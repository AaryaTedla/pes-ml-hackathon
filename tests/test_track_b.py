"""Track B grouping, candidate scoring and inference contract checks."""
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.train_models import (FEATURES, KEYS, cap_points, fit_model,
                              make_model, predict_chunks, score)


def fixture():
    rng = np.random.default_rng(42)
    rows = []
    for trip in range(6):
        for point in range(12):
            for arc in range(3):
                row = dict(trip_id=f'mock_{trip}', point_idx=point,
                           arc_id=point*3+arc, label=int(arc == point % 3))
                row.update({feature: float(rng.uniform()) for feature in FEATURES})
                rows.append(row)
    return pd.DataFrame(rows)


class TrackBTests(unittest.TestCase):
    def test_fit_cap_retains_complete_point_groups(self):
        frame = fixture()
        capped = cap_points(frame, 50, 42)
        self.assertLessEqual(len(capped), 50)
        self.assertTrue(capped.groupby(KEYS[:2]).size().eq(3).all())
        pd.testing.assert_frame_equal(capped, cap_points(frame, 50, 42))

    def test_no_trip_crosses_outer_fold(self):
        frame = fixture()
        for fit, valid in GroupKFold(3).split(frame, groups=frame.trip_id):
            capped = cap_points(frame.iloc[fit], 50, 42)
            self.assertFalse(set(capped.trip_id) & set(frame.iloc[valid].trip_id))

    def test_chunked_pipeline_matches_direct_probability(self):
        frame = fixture()
        model, _, _ = fit_model(make_model('LogReg'), 'LogReg', frame, None, 42)
        chunked = predict_chunks(model, frame, chunk=7)
        expected = model.predict_proba(frame[FEATURES])[:, list(model.classes_).index(1)]
        np.testing.assert_allclose(chunked, expected)

    def test_missing_positive_is_top1_error(self):
        frame = fixture().iloc[:3].copy()
        frame['label'] = 0
        result = score(frame, np.array([.1, .2, .3]))
        self.assertEqual(result['top1'], 0)
        self.assertEqual(result['points'], 1)


if __name__ == '__main__':
    unittest.main()
