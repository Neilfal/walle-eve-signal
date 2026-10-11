import math
import unittest

import numpy as np

from walle_eve.belief import Belief
from walle_eve.config import DEFAULT_SENSOR
from walle_eve.geometry import angle_diff, bearing_deg, compass_word, euclid


def make_belief():
    return Belief(np.zeros((15, 15), dtype=bool), dict(DEFAULT_SENSOR))


class GeometryTests(unittest.TestCase):
    def test_bearings(self):
        self.assertAlmostEqual(bearing_deg((5, 5), (5, 2)), 0.0)      # north = up
        self.assertAlmostEqual(bearing_deg((5, 5), (8, 5)), 90.0)     # east
        self.assertAlmostEqual(bearing_deg((5, 5), (5, 9)), 180.0)    # south
        self.assertAlmostEqual(bearing_deg((5, 5), (2, 5)), 270.0)    # west
        self.assertEqual(compass_word(45), "NE")

    def test_angle_diff_wraps(self):
        self.assertAlmostEqual(float(angle_diff(350.0, 10.0)), -20.0)
        self.assertAlmostEqual(float(angle_diff(10.0, 350.0)), 20.0)


class BeliefTests(unittest.TestCase):
    def test_starts_uniform_and_normalised(self):
        b = make_belief()
        self.assertAlmostEqual(float(b.p.sum()), 1.0)
        self.assertAlmostEqual(b.entropy(), math.log2(225), places=5)

    def test_walls_get_zero_probability(self):
        walls = np.zeros((6, 6), dtype=bool)
        walls[2:4, 2:4] = True
        b = Belief(walls, dict(DEFAULT_SENSOR))
        b.update_strength((0, 0), 0.4)
        self.assertEqual(float(b.p[walls].sum()), 0.0)
        self.assertAlmostEqual(float(b.p.sum()), 1.0)

    def test_evidence_concentrates_belief_on_truth(self):
        eve = (11, 4)
        sensor = dict(DEFAULT_SENSOR)
        b = Belief(np.zeros((15, 15), dtype=bool), sensor)
        h0 = b.entropy()
        for pos in [(1, 1), (1, 12), (8, 8), (13, 13)]:      # noiseless readings from 4 places
            d = euclid(pos, eve)
            b.update_strength(pos, sensor["s0"] * math.exp(-d / sensor["decay_length"]))
            b.update_bearing(pos, bearing_deg(pos, eve))
        self.assertLess(b.entropy(), h0 - 4)
        self.assertEqual(b.estimate(), eve)
        self.assertGreater(b.confidence(), 0.9)

    def test_outlier_floor_survives_a_wrong_reading(self):
        b = make_belief()
        b.update_clue_bearing((7, 7), 0.0, 10.0)             # says: due north of (7,7)
        self.assertTrue((b.p > 0).all())                     # nothing is ever ruled out completely

    def test_visual_negative_and_positive(self):
        b = make_belief()
        b.update_visual((7, 7), seen=False)
        self.assertLess(float(b.p[7, 7]), 0.001)
        b.update_visual((3, 3), seen=True, eve_pos=(4, 3))
        self.assertEqual(b.estimate(), (4, 3))
        self.assertAlmostEqual(b.confidence(), 1.0)


if __name__ == "__main__":
    unittest.main()
