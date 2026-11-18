import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hmm import decode, mock_probabilities


class HMMTests(unittest.TestCase):
    def trip(self):
        return dict(track=np.array([[1., 0., 0.], [2., 0., 1.]]), arcs=np.array([[0, 1], [2, 3]]))

    def test_transition_and_gap(self):
        trip = self.trip()
        rows = pd.DataFrame(dict(point_idx=[0, 0, 1, 1], arc_id=[0, 1, 0, 1], prob=[.99, .01, .4, .6]))
        result = decode(trip, rows, {})
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [0, 0])
        trip['track'][1, 2] = 100.
        np.testing.assert_array_equal(decode(trip, rows, {}), [0, 1])

    def test_missing_points_and_invalid_probabilities(self):
        rows = pd.DataFrame(dict(point_idx=[1], arc_id=[0], prob=[0.]))
        np.testing.assert_array_equal(decode(self.trip(), rows, {}), [-1, 0])
        rows['prob'] = np.nan
        with self.assertRaises(ValueError):
            decode(self.trip(), rows, {})

    def test_mock_softmax(self):
        rows = pd.DataFrame(dict(point_idx=[0, 0, 1], dist_m=[2., 2., 1e8], cos_head=[1., -1., 0.]))
        result = mock_probabilities(rows)
        np.testing.assert_allclose(result.groupby('point_idx').prob.sum(), 1.)
        self.assertGreater(result.prob[0], result.prob[1])

    def test_bounded_network_mode(self):
        trip = dict(nodes=np.array([[0., 0.], [.0001, 0.], [.0002, 0.]]),
                    arcs=np.array([[0, 1], [1, 2]]),
                    track=np.array([[.00002, 0., 0.], [.00015, 0., 1.]]))
        rows = pd.DataFrame(dict(point_idx=[0, 0, 1, 1], arc_id=[0, 1, 0, 1],
                                 prob=[.99, .01, .1, .9]))
        before = rows.copy(deep=True)
        np.testing.assert_array_equal(decode(trip, rows, {'mode': 'network'}), [0, 1])
        pd.testing.assert_frame_equal(rows, before)


if __name__ == '__main__':
    unittest.main()
