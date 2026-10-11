"""Run one mission end to end: environment + agent + trace + report + files."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .agent import Agent
from .config import load_config
from .env import MissionEnv
from .report import build_report, report_markdown
from .trace import Trace, _jsonable
from .viz import LEGEND, ascii_frames, plot_summary, render_ascii


def run_mission(config, out_dir: Optional[str] = None, env_seed: Optional[int] = None,
                agent_seed: int = 0, agent_params: Optional[Dict[str, Any]] = None,
                make_plots: bool = True, verbose: bool = False, max_steps: int = 2000) -> Dict[str, Any]:
    """``config`` is a dict or a path to a .json/.yaml file. Returns report, trace, agent and env."""
    if isinstance(config, str):
        config = load_config(config)
    env = MissionEnv(config, seed=env_seed)
    agent = Agent(env.briefing(), seed=agent_seed, params=agent_params)
    trace = Trace()

    obs = env.reset()
    agent.begin(obs)
    while not env.done and env.steps < max_steps:
        pos_before, batt_before = agent.pos, agent.battery
        action, reasoning, meta = agent.act()
        obs = env.step(action)
        agent.observe(action, obs)
        trace.log(step=obs["step"], time_used=obs["time_used"], mode=meta.get("mode", agent.mode),
                  position_before=list(pos_before), battery_before=batt_before, action=action,
                  reasoning=reasoning, plan=meta or None, observation=obs,
                  belief={"estimate": list(agent.confirmed or agent.belief.estimate()),
                          "confidence": 1.0 if agent.confirmed else agent.belief.confidence(),
                          "entropy_bits": agent.belief.entropy()})
        if verbose:
            print(Trace.format_entry(trace.entries[-1]))

    report = build_report(agent, env)
    result = {"report": report, "trace": trace, "agent": agent, "env": env}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        trace.write_jsonl(os.path.join(out_dir, "trace.jsonl"))
        trace.write_text(os.path.join(out_dir, "trace.txt"), title=f"Mission trace: {env.name}")
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=_jsonable)
        with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
            fh.write(report_markdown(report))
        with open(os.path.join(out_dir, "map_frames.txt"), "w", encoding="utf-8") as fh:
            fh.write(ascii_frames(agent, env))
        with open(os.path.join(out_dir, "map_final.txt"), "w", encoding="utf-8") as fh:
            fh.write(LEGEND + "\n\n" + render_ascii(env.walls, env.chargers, agent.frames[-1],
                     tuple(env.ground_truth()["eve_position"]), title=f"Final map: {env.name}") + "\n")
        if make_plots:
            plot_summary(agent, env, os.path.join(out_dir, "mission.png"))
    return result
