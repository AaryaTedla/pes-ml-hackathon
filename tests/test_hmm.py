import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hmm import decode, mock_probabilities, HMMConfig


class TinyNet:
    node_xy = np.array([[0., 0.], [10., 0.], [1000., 0.], [1010., 0.]])
    arcs = np.array([[0, 1], [2, 3]])
    A = node_xy[arcs[:, 0]]
    B = node_xy[arcs[:, 1]]
    length = np.array([10., 10.])

    def project_points(self, x, y):
        return np.column_stack([x, y])


class HMMTests(unittest.TestCase):
    def test_transition_overcomes_independent_wrong_choice_and_gap_resets(self):
        trip = dict(track=np.array([[1., 0., 0.], [2., 0., 1.]]))
        rows = dict(point_idx=[0, 0, 1, 1], arc_id=[0, 1, 0, 1],
                    probability=[.99, .01, .4, .6])
        self.assertEqual(decode(trip, rows, net=TinyNet()), [0, 0])
        trip['track'][1, 2] = 100.
        self.assertEqual(decode(trip, rows, net=TinyNet()), [0, 1])

    def test_tiny_probabilities_and_invalid_input(self):
        trip = dict(track=np.array([[1., 0., 0.], [2., 0., 1.]]))
        rows = dict(point_idx=[0, 1], arc_id=[0, 0], probability=[1e-300, 0.])
        self.assertEqual(decode(trip, rows, net=TinyNet()), [0, 0])
        rows['probability'] = [np.nan, .5]
        with self.assertRaises(ValueError):
            decode(trip, rows, net=TinyNet())

    def test_mock_normalises_per_point_and_prefers_heading(self):
        rows = pd.DataFrame(dict(point_idx=[0, 0, 1], dist_m=[2., 2., 1e8],
                                 cos_head=[1., -1., 0.]))
        scored = mock_probabilities(rows, HMMConfig())
        np.testing.assert_allclose(scored.groupby('point_idx').probability.sum(), 1.)
        self.assertGreater(scored.probability[0], scored.probability[1])


if __name__ == '__main__':
    unittest.main()
