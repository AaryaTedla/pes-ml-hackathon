"""Budgeted PathSnap comparison. Only training trips are sampled; test is untouched."""
import copy
import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = ['dist_m', 'dist_rel', 'cand_rank', 'n_cand', 'proj_frac', 'arc_len_m',
            'cos_head', 'abs_head_diff_deg', 'speed_mps', 'moving', 'dist_prev1',
            'dist_next1', 'dist_prev2', 'dist_next2', 'dist_win_mean',
            'from_out_deg', 'to_out_deg', 'dt_prev_s']
KEYS = ['trip_id', 'point_idx', 'arc_id']
SEED = 42
RULE = ('Choose highest mean per-fold CV per-point Top-1; exact ties use lower '
        'mean CV candidate log loss. Student-test is comparison only. '
        'Compare against nearest-arc on identical scopes. Baseline failure is '
        'a diagnostic flag, not proof of an implementation error.')
GROUPS = {'fast': ['LogReg', 'LightGBM'],
          'slow': ['RandomForest', 'MLP', 'KNN'],
          'all': ['LogReg', 'LightGBM', 'RandomForest', 'MLP', 'KNN']}
CAPS = {'LogReg': None, 'LightGBM': None, 'RandomForest': 300000,
        'MLP': 300000, 'KNN': 150000}


def load_tables(train_path, test_path, n_trips=40):
    train, test = pd.read_parquet(train_path), pd.read_parquet(test_path)
    for name, frame in [('train', train), ('student_test', test)]:
        missing = set(FEATURES + KEYS + ['label', 'pt_has_pos']) - set(frame)
        if missing:
            raise ValueError(f'{name}: missing {missing}')
        frame['trip_id'] = frame.trip_id.astype(str)
        if frame[KEYS + ['label', 'pt_has_pos']].isna().any().any():
            raise ValueError(f'{name}: null identifiers or labels')
        if frame.duplicated(KEYS).any() or not frame.label.isin([0, 1]).all():
            raise ValueError(f'{name}: duplicate candidates or invalid labels')
        if not frame.pt_has_pos.isin([0, 1]).all():
            raise ValueError(f'{name}: invalid filter values')
        actual = frame.groupby(KEYS[:2]).label.transform('sum')
        if not (actual == frame.pt_has_pos).all():
            raise ValueError(f'{name}: point filter disagrees with labels')
        frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    if train.trip_id.nunique() != 80 or test.trip_id.nunique() != 10:
        raise ValueError('Expected 80 train and 10 student-test trips')
    if set(train.trip_id) & set(test.trip_id):
        raise ValueError('Trip leakage')
    selected = sorted(np.random.default_rng(SEED).choice(
        sorted(train.trip_id.unique()), size=n_trips, replace=False).tolist())
    train = train.loc[train.trip_id.isin(selected) & train.pt_has_pos.eq(1)].reset_index(drop=True)
    return train, test.reset_index(drop=True), selected


def cap_points(frame, cap, seed):
    """Sample complete observation candidate sets, only within the fit partition."""
    if cap is None or len(frame) <= cap:
        return frame.reset_index(drop=True)
    sizes = frame.groupby(KEYS[:2], sort=False).size().reset_index(name='rows')
    order = np.random.default_rng(seed).permutation(len(sizes))
    shuffled = sizes.iloc[order]
    keep = shuffled.loc[shuffled.rows.cumsum() <= cap, KEYS[:2]]
    result = frame.merge(keep, on=KEYS[:2], how='inner', sort=False)
    if result.empty:
        raise ValueError('Training cap too small')
    return result.reset_index(drop=True)


def make_model(name, smoke=False):
    estimators = {
        'LogReg': LogisticRegression(max_iter=200, random_state=SEED, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=5 if smoke else 300,
            learning_rate=.1, num_leaves=63, random_state=SEED, n_jobs=-1,
            verbosity=-1, deterministic=True, force_col_wise=True),
        'RandomForest': RandomForestClassifier(n_estimators=5 if smoke else 100,
            max_depth=18, min_samples_leaf=20, random_state=SEED, n_jobs=-1),
        # The fit function implements early stopping using a whole-trip holdout.
        # A native MLP random row validation split would violate that requirement.
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=2048,
            max_iter=3 if smoke else 30, early_stopping=False, random_state=SEED),
        'KNN': KNeighborsClassifier(n_neighbors=25, weights='distance',
                                    algorithm='ball_tree', n_jobs=-1),
    }
    return Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scaler', StandardScaler() if name in ['LogReg', 'KNN', 'MLP'] else 'passthrough'),
                     ('model', estimators[name])])


def fit_model(pipeline, name, frame, cap, seed, params=None):
    frame = cap_points(frame, cap, seed)
    if set(frame.label) != {0, 1}:
        raise ValueError('Training needs both classes')
    if params:
        pipeline.set_params(**{f'model__{k}': v for k, v in params.items()})
    history = []
    start = perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
        warnings.simplefilter('always')
        if name == 'MLP':
            trip_ids = sorted(frame.trip_id.unique())
            if len(trip_ids) < 3:
                raise ValueError('MLP grouped early stopping needs at least three fit trips')
            holdout = np.random.default_rng(seed).choice(trip_ids,
                size=max(1, len(trip_ids)//5), replace=False)
            fit = frame.loc[~frame.trip_id.isin(holdout)]
            valid = frame.loc[frame.trip_id.isin(holdout)]
            prep = clone(pipeline[:-1])
            x = prep.fit_transform(fit[FEATURES])
            xv = prep.transform(valid[FEATURES])
            estimator = clone(pipeline.named_steps['model'])
            best_loss, best_epoch, stale = float('inf'), 1, 0
            for epoch in range(estimator.max_iter):
                estimator.partial_fit(x, fit.label, classes=np.array([0, 1]))
                loss = log_loss(valid.label, estimator.predict_proba(xv), labels=[0, 1])
                history.append(dict(epoch=epoch+1, training_loss=estimator.loss_,
                                    validation_log_loss=loss))
                if loss < best_loss - 1e-4:
                    best_loss, best_epoch, stale = loss, epoch+1, 0
                else:
                    stale += 1
                if stale >= 5:
                    break
            pipeline.set_params(model__max_iter=best_epoch)
        pipeline.fit(frame[FEATURES], frame.label)
    return pipeline, dict(fit_seconds=perf_counter()-start, fitted_rows=len(frame),
        fitted_trips=frame.trip_id.nunique(), warnings=' | '.join(str(w.message) for w in caught)), history


def predict_chunks(pipeline, frame, chunk=4096):
    output = np.empty(len(frame))
    column = list(pipeline.classes_).index(1)
    with threadpool_limits(limits=4):
        for start in range(0, len(frame), chunk):
            stop = min(start+chunk, len(frame))
            output[start:stop] = pipeline.predict_proba(frame.iloc[start:stop][FEATURES])[:, column]
    return output


def score(frame, probabilities):
    predictions = probabilities >= .5
    ranked = frame[KEYS[:2] + ['label']].copy().reset_index(drop=True)
    ranked['p'] = probabilities
    chosen = ranked.loc[ranked.groupby(KEYS[:2], sort=False).p.idxmax()]
    return dict(top1=float(chosen.label.mean()),
        macro_f1=f1_score(frame.label, predictions, labels=[0, 1], average='macro', zero_division=0),
        precision=precision_score(frame.label, predictions, labels=[0, 1], average='macro', zero_division=0),
        recall=recall_score(frame.label, predictions, labels=[0, 1], average='macro', zero_division=0),
        log_loss=log_loss(frame.label, np.column_stack([1-probabilities, probabilities]), labels=[0, 1]),
        confusion_matrix=json.dumps(confusion_matrix(frame.label, predictions, labels=[0, 1]).tolist()),
        rows=len(frame), points=len(chosen))


def nearest_top1(frame):
    ranked = frame.reset_index(drop=True)
    indices = ranked.groupby(KEYS[:2], sort=False).cand_rank.idxmin()
    return float(ranked.loc[indices].label.mean())


def evaluate(pipeline, frame):
    start = perf_counter()
    probabilities = predict_chunks(pipeline, frame)
    seconds = perf_counter()-start
    return dict(**score(frame, probabilities), inference_seconds=seconds,
                baseline_top1=nearest_top1(frame)), probabilities


def compare(train, test, selected, out, group='fast', smoke=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    splits = list(GroupKFold(3).split(train, train.label, train.trip_id))
    fold_rows = []
    for fold, (fit, valid) in enumerate(splits):
        assert not set(train.iloc[fit].trip_id) & set(train.iloc[valid].trip_id)
        fold_rows.extend(dict(trip_id=t, fold=fold) for t in train.iloc[valid].trip_id.unique())
    pd.DataFrame(fold_rows).to_csv(out/'fold_assignments.csv', index=False)
    metadata = dict(selection_rule=RULE, seed=SEED, selected_trips=selected, folds=3,
        features=FEATURES, group=group, synthetic=smoke, train_rows=len(train),
        student_test_rows=len(test), caps=CAPS,
        mlp_stopping='Whole-trip holdout inside fitting data; refit for best epoch count. Native early_stopping=False.',
        versions={p:version(p) for p in ['numpy','pandas','scikit-learn','lightgbm','joblib']},
        python=platform.python_version())
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    rows, summary = [], []
    for name in GROUPS[group]:
        params = make_model(name, smoke).named_steps['model'].get_params()
        (out/f'{name}_config.json').write_text(json.dumps(params,indent=2,default=str))
        print(f'{name}: starting capped 3-fold run', flush=True)
        for fold, (fit, valid) in enumerate(splits):
            model, timings, history = fit_model(make_model(name, smoke), name,
                train.iloc[fit], CAPS[name], SEED+fold)
            result, _ = evaluate(model, train.iloc[valid])
            rows.append(dict(model=name, partition='cv', fold=fold, **timings, **result))
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            print(f'{name} fold={fold}: Top-1={result["top1"]:.4f}, log loss={result["log_loss"]:.4f}', flush=True)
        model, timings, history = fit_model(make_model(name, smoke), name, train, CAPS[name], SEED)
        result, probabilities = evaluate(model, test)  # ALL student-test rows, never capped
        rows.append(dict(model=name, partition='student_test', fold=-1, **timings, **result))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        if not smoke:
            joblib.dump(model,out/f'{name}.joblib')
        pd.DataFrame({'training_loss':getattr(model.named_steps['model'],'loss_curve_',[])}).to_csv(
            out/f'{name}_loss_curve.csv',index=False)
        if history:
            pd.DataFrame(history).to_csv(out/'MLP_grouped_early_stopping.csv',index=False)
        scores = test[KEYS].copy(); scores['probability_correct'] = probabilities
        scores.to_parquet(out/f'{name}_student_scores.parquet',index=False)
        cv = pd.DataFrame([r for r in rows if r['model']==name and r['partition']=='cv'])
        entry = dict(model=name, protocol='40-trip-3fold-capped-v1', seed=SEED)
        for column in ['top1','macro_f1','precision','recall','log_loss','fit_seconds','inference_seconds','baseline_top1']:
            entry[f'cv_{column}'] = float(cv[column].mean())
            entry[f'test_{column}'] = result[column] if column in result else timings[column]
        entry['cv_top1_std'] = float(cv.top1.std())
        entry['beats_nearest_cv'] = entry['cv_top1'] > entry['cv_baseline_top1']
        entry['beats_nearest_test'] = entry['test_top1'] > entry['test_baseline_top1']
        entry['selected_trips'] = json.dumps(selected)
        summary.append(entry)
        pd.DataFrame(summary).to_csv(out/'comparison.csv',index=False)
        print(f'{name} done: student Top-1={result["top1"]:.4f}',flush=True)
    return pd.DataFrame(summary)


def merge_comparisons(paths):
    table = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if table.model.duplicated().any():
        raise ValueError('Duplicate model results: pass only one result file per group')
    if table.selected_trips.nunique()!=1 or table.protocol.nunique()!=1 or table.seed.nunique()!=1:
        raise ValueError('Comparison protocols or training-trip subsets differ')
    if set(table.model) != set(GROUPS['all']):
        raise ValueError('Need all five models before choosing a winner')
    return table.sort_values(['cv_top1','cv_log_loss'],ascending=[False,True]).reset_index(drop=True)


def tune_winner(train, test, name, out, trials=25, timeout=600, smoke=False):
    import optuna
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    splits=list(GroupKFold(3).split(train,train.label,train.trip_id))
    def search(trial):
        if name=='LightGBM':
            return dict(num_leaves=trial.suggest_int('num_leaves',15,95),
                learning_rate=trial.suggest_float('learning_rate',.02,.2,log=True),
                min_child_samples=trial.suggest_int('min_child_samples',10,100),
                feature_fraction=trial.suggest_float('feature_fraction',.6,1),
                lambda_l2=trial.suggest_float('lambda_l2',1e-4,10,log=True),
                n_estimators=trial.suggest_int('n_estimators',100,400,step=100))
        if name=='LogReg':return {'C':trial.suggest_float('C',.001,100,log=True)}
        if name=='RandomForest':return dict(max_depth=trial.suggest_int('max_depth',10,22),min_samples_leaf=trial.suggest_int('min_samples_leaf',10,50))
        if name=='MLP':return dict(alpha=trial.suggest_float('alpha',1e-6,.1,log=True),learning_rate_init=trial.suggest_float('learning_rate_init',1e-4,.01,log=True))
        return dict(n_neighbors=trial.suggest_int('n_neighbors',15,35,step=2))
    def objective(trial):
        params=search(trial);losses=[];top1=[]
        for fold,(fit,valid) in enumerate(splits):
            model,_,_=fit_model(make_model(name,smoke),name,train.iloc[fit],CAPS[name],SEED+fold,params)
            result,_=evaluate(model,train.iloc[valid])
            losses.append(result['log_loss']);top1.append(result['top1'])
        trial.set_user_attr('fold_log_loss',losses)
        trial.set_user_attr('mean_top1',float(np.mean(top1)))
        return float(np.mean(losses))
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED,n_startup_trials=5),
        storage=f'sqlite:///{(out/"study.sqlite3").resolve()}',study_name=name)
    start=perf_counter()
    try:
        study.optimize(objective,n_trials=trials,timeout=timeout,
            callbacks=[lambda s,t:s.trials_dataframe().to_csv(out/'optuna_trials.csv',index=False)])
    finally:study.trials_dataframe().to_csv(out/'optuna_trials.csv',index=False)
    search_seconds=perf_counter()-start
    comparisons=[]
    best=None
    for label,params in [('untuned',None),('tuned',study.best_params)]:
        model,timings,history=fit_model(make_model(name,smoke),name,train,CAPS[name],SEED,params)
        result,_=evaluate(model,test)
        comparisons.append(dict(model=name,version=label,**timings,**result))
        if label=='tuned':best=model
    pd.DataFrame(comparisons).to_csv(out/'tuned_vs_untuned.csv',index=False)
    if not smoke:
        joblib.dump(best,out/'model.joblib')
        joblib.dump(best.named_steps['scaler'],out/'scaler.joblib')
        joblib.dump(best.named_steps['imputer'],out/'imputer.joblib')
    (out/'feature_columns.json').write_text(json.dumps(FEATURES,indent=2))
    (out/'tuning_metadata.json').write_text(json.dumps(dict(model=name,seed=SEED,
        requested_trials=trials,completed_trials=len(study.trials),timeout_seconds=timeout,
        search_seconds=search_seconds,best_params=study.best_params,
        objective='3-fold CV candidate log loss; minimize',best_cv_log_loss=study.best_value,
        note='Timeout is checked between trials; an active trial and final refits may exceed it.'),indent=2))
    return pd.DataFrame(comparisons)


def plot_comparison(table, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for metric in ['top1','macro_f1','log_loss']:
        table.set_index('model')[[f'cv_{metric}',f'test_{metric}']].plot.bar(rot=25)
        plt.ylabel(metric);plt.tight_layout();plt.savefig(output/f'{metric}.png',dpi=150);plt.close()
