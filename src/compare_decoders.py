"""Compare classifier-only vs the two HMM decoders on identical whole trips and identical probabilities.

    python src/compare_decoders.py --prob mock     # stand-in probabilities
    python src/compare_decoders.py --prob model    # artifacts/model.joblib probabilities (once 496's model exists)

Splits are the same seeded whole-trip TUNE / VAL splits as eval_hmm.py, plus the 10 student_test trips.
Decoders: hmm.py (multi-hop, artifacts/hmm_config.json) and hmm_fixed.py (fixed transitions, artifacts/hmm_fixed_config.json).
Output: results/hmm_decoder_comparison_<prob>.csv (per split) and ..._per_trip.csv. Evaluator trips are never used.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import eval_hmm
import hmm, hmm_fixed

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", default="mock", choices=["mock", "model"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    D = eval_hmm.load_data(a.prob)
    tune, val, test = eval_hmm.make_splits(D, a.seed)
    load = lambda n: json.load(open(os.path.join(ROOT, "artifacts", n)))
    decoders = (("multi-hop (hmm.py)", hmm.decode, load("hmm_config.json")),
                ("fixed-transition (hmm_fixed.py)", hmm_fixed.decode, load("hmm_fixed_config.json")))
    rows, per_trip = [], []
    for name, fn, cfg in decoders:
        eval_hmm.decode = fn
        for split, keys in (("TUNE", tune), ("VAL", val), ("STUDENT_TEST", test)):
            df = eval_hmm.eval_set(D, keys, cfg).assign(decoder=name, split=split)
            per_trip.append(df)
            rows.append(dict(prob_source=a.prob, decoder=name, split=split, trips=len(df),
                             acc_classifier_only=df.acc_clf.mean(), acc_hmm=df.acc_hmm.mean(),
                             continuity_classifier_only=df.cont_clf.mean(), continuity_hmm=df.cont_hmm.mean(),
                             trips_better=int((df.acc_hmm > df.acc_clf).sum()),
                             trips_worse=int((df.acc_hmm < df.acc_clf - 0.005).sum()),
                             hmm_seconds=df.hmm_runtime_s.sum()))
    out = pd.DataFrame(rows).round(4)
    out.to_csv(os.path.join(ROOT, "results", f"hmm_decoder_comparison_{a.prob}.csv"), index=False)
    pd.concat(per_trip).to_csv(os.path.join(ROOT, "results", f"hmm_decoder_comparison_{a.prob}_per_trip.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
