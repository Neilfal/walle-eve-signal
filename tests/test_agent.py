import json
import os
import tempfile
import unittest

from walle_eve import Agent, MissionEnv, run_mission
from walle_eve.generator import random_config

HERE = os.path.dirname(__file__)
CONFIGS = os.path.join(HERE, "..", "configs")


class AgentTests(unittest.TestCase):
    def test_sample_missions_succeed_and_never_strand(self):
        for name in ("sample1.json", "sample2_decoys.json"):
            res = run_mission(os.path.join(CONFIGS, name), make_plots=False)
            r = res["report"]
            self.assertTrue(r["evaluation_vs_ground_truth"]["located_within_visual_range"], name)
            self.assertFalse(r["resources"]["stranded"], name)

    def test_agent_always_keeps_a_way_home(self):
        """Safety property: on random missions WALL-E is never left stranded away from a charger."""
        for seed in range(30, 38):
            cfg = random_config(seed, battery=30, n_decoys=1, spoof_prob=0.05)
            res = run_mission(cfg, make_plots=False, agent_seed=seed)
            self.assertNotEqual(res["env"].done_reason, "out_of_battery", f"seed {seed}")
            self.assertEqual(res["env"].invalid_count, 0, f"seed {seed}: agent sent an invalid action")
            self.assertLessEqual(res["env"].time_used, cfg["budget"]["time"])

    def test_agent_recharges_when_battery_is_tight(self):
        res = run_mission(os.path.join(CONFIGS, "sample1.json"), make_plots=False)
        self.assertGreaterEqual(res["report"]["resources"]["recharges"], 1)
        modes = [t["to"] for t in res["report"]["strategy_timeline"]]
        self.assertIn("return", modes)
        self.assertIn("recharge", modes)

    def test_deterministic_given_seeds(self):
        cfg = os.path.join(CONFIGS, "sample2_decoys.json")
        a = run_mission(cfg, make_plots=False)["report"]
        b = run_mission(cfg, make_plots=False)["report"]
        self.assertEqual(a["inferred_location"], b["inferred_location"])
        self.assertEqual(a["resources"], b["resources"])

    def test_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as d:
            run_mission(os.path.join(CONFIGS, "sample1.json"), out_dir=d, make_plots=False)
            for f in ("trace.jsonl", "trace.txt", "report.json", "report.md", "map_frames.txt", "map_final.txt"):
                self.assertTrue(os.path.getsize(os.path.join(d, f)) > 0, f)
            with open(os.path.join(d, "trace.jsonl")) as fh:
                first = json.loads(fh.readline())
            for key in ("action", "reasoning", "observation", "belief"):
                self.assertIn(key, first)
            with open(os.path.join(d, "report.json")) as fh:
                rep = json.load(fh)
            for key in ("evidence", "inferred_location", "confidence", "recommended_next_step"):
                self.assertIn(key, rep)

    def test_clue_consistency_check(self):
        """A clue that contradicts the final belief is flagged; an honest one is not."""
        env = MissionEnv(random_config(1))
        agent = Agent(env.briefing())
        agent.belief.update_visual((9, 9), seen=True, eve_pos=(10, 10))      # belief: EVE at (10, 10)
        honest = {"type": "bearing", "position": [4, 10], "bearing_deg": 92.0, "sigma": 15.0}
        decoy = {"type": "bearing", "position": [4, 10], "bearing_deg": 0.0, "sigma": 15.0}
        near = {"type": "range", "position": [4, 10], "distance": 6.5, "sigma": 1.5}
        far = {"type": "range", "position": [4, 10], "distance": 12.5, "sigma": 1.5}
        self.assertEqual(agent.clue_consistency(honest), "consistent")
        self.assertEqual(agent.clue_consistency(decoy), "conflicting")
        self.assertEqual(agent.clue_consistency(near), "consistent")
        self.assertEqual(agent.clue_consistency(far), "conflicting")
        self.assertEqual(agent.clue_consistency({"type": "status", "position": [4, 10]}), "n/a")

    def test_decoys_and_spoofs_do_not_break_the_search(self):
        r = run_mission(os.path.join(CONFIGS, "sample2_decoys.json"), make_plots=False)["report"]
        ev = r["evaluation_vs_ground_truth"]
        self.assertTrue(ev["decoy_clues_found"])
        self.assertGreater(ev["spoofed_scans_received"], 0)
        self.assertTrue(ev["located_within_visual_range"])

if __name__ == "__main__":
    unittest.main()
