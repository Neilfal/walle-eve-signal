import copy
import json
import os
import unittest

from walle_eve.config import ConfigError, normalize_config
from walle_eve.env import MissionEnv

HERE = os.path.dirname(__file__)
SAMPLE = os.path.join(HERE, "..", "configs", "sample1.json")


def load():
    with open(SAMPLE) as fh:
        return json.load(fh)


class ConfigTests(unittest.TestCase):
    def test_sample_is_valid(self):
        cfg = normalize_config(load())
        self.assertEqual(cfg["grid"]["width"], 15)

    def test_missing_key_rejected(self):
        cfg = load()
        del cfg["budget"]
        with self.assertRaises(ConfigError):
            normalize_config(cfg)

    def test_wall_start_rejected(self):
        cfg = load()
        cfg["start"] = [4, 2]            # a wall in sample1
        with self.assertRaises(ConfigError):
            normalize_config(cfg)

    def test_unreachable_eve_rejected(self):
        cfg = {"grid": {"rows": ["..#..", "..#..", "..#.."]}, "start": [0, 0], "chargers": [[0, 0]],
               "eve": {"position": [4, 1]}, "initial_signal": {"bearing_deg": 90, "strength": 0.3},
               "budget": {"battery": 20, "time": 50}}
        with self.assertRaises(ConfigError):
            normalize_config(cfg)


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.env = MissionEnv(load())

    def test_briefing_hides_secrets(self):
        b = json.dumps(self.env.briefing())
        for secret in ("eve", "clues", "spoof_prob", "decoy", "low_power"):
            self.assertNotIn(f'"{secret}"', b)
        self.assertNotIn("[13, 12]", b)

    def test_move_costs_and_walls(self):
        obs = self.env.step({"type": "move", "dir": "E"})
        self.assertTrue(obs["ok"])
        self.assertEqual(obs["position"], [2, 1])
        self.assertEqual(obs["battery"], self.env.budget["battery"] - 1)
        env = MissionEnv(load())
        env.step({"type": "move", "dir": "N"})            # (1,1) -> (1,0): top edge
        obs = env.step({"type": "move", "dir": "N"})      # off the map
        self.assertFalse(obs["ok"])
        self.assertEqual(obs["event"], "blocked")
        self.assertEqual(obs["battery"], env.budget["battery"] - 1)   # only the first move was charged

    def test_recharge_only_at_charger(self):
        self.env.step({"type": "move", "dir": "E"})
        self.assertFalse(self.env.step({"type": "recharge"})["ok"])
        self.env.step({"type": "move", "dir": "W"})
        obs = self.env.step({"type": "recharge"})
        self.assertTrue(obs["ok"])
        self.assertEqual(obs["battery"], self.env.budget["battery"])
        self.assertEqual(obs["recharges_left"], self.env.budget["max_recharges"] - 1)

    def test_scan_reveals_clues_only_in_radius(self):
        far = self.env.step({"type": "scan"})
        self.assertEqual(far["scan"]["clues"], [])
        cfg = load()
        cfg["start"] = [6, 4]            # clue c1 sits at (6, 3)
        env = MissionEnv(cfg)
        first = env.step({"type": "scan"})["scan"]["clues"]
        self.assertEqual([c["id"] for c in first], ["c1"])
        self.assertFalse(first[0]["already_known"])
        again = env.step({"type": "scan"})["scan"]["clues"]
        self.assertTrue(again[0]["already_known"])
        self.assertEqual(first[0]["text"], again[0]["text"])

    def test_sighting_within_visual_range(self):
        cfg = load()
        cfg["start"] = [13, 11]
        env = MissionEnv(cfg)
        scan = env.step({"type": "scan"})["scan"]
        self.assertTrue(scan["eve_in_view"])
        self.assertEqual(scan["eve_position"], [13, 12])

    def test_budget_exhaustion_ends_mission(self):
        cfg = load()
        cfg["budget"]["battery"] = 3
        cfg["budget"]["max_recharges"] = 0
        env = MissionEnv(cfg)
        env.step({"type": "scan"})
        obs = env.step({"type": "move", "dir": "E"})
        self.assertTrue(obs["done"])
        self.assertEqual(obs["done_reason"], "out_of_battery")

    def test_same_seed_same_world(self):
        a, b = MissionEnv(load(), seed=5), MissionEnv(load(), seed=5)
        sa, sb = a.step({"type": "scan"}), b.step({"type": "scan"})
        self.assertEqual(sa["scan"], sb["scan"])


if __name__ == "__main__":
    unittest.main()
