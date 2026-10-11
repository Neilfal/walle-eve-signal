"""Full action trace: what the agent did, saw, decided and why."""
from __future__ import annotations

import json
from typing import Any, Dict, List


def _jsonable(o):
    if hasattr(o, "item"):          # numpy scalar
        return o.item()
    if hasattr(o, "tolist"):        # numpy array
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


class Trace:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def log(self, **entry: Any) -> None:
        self.entries.append(entry)

    # ------------------------------------------------------------------ output
    def write_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for e in self.entries:
                fh.write(json.dumps(e, default=_jsonable) + "\n")

    def write_text(self, path: str, title: str = "Mission trace") -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"{title}\n{'=' * len(title)}\n\n")
            for e in self.entries:
                fh.write(self.format_entry(e) + "\n")

    @staticmethod
    def format_entry(e: Dict[str, Any]) -> str:
        a, o, b = e["action"], e["observation"], e["belief"]
        what = a["type"].upper() + (f" {a['dir']}" if a["type"] == "move" else "")
        head = (f"[step {e['step']:03d} | t={e['time_used']:03d} | battery {e['battery_before']:g} -> {o['battery']:g}"
                f" | mode {e['mode']}] {what} from {tuple(e['position_before'])}")
        lines = [head, f"   why : {e['reasoning']}", f"   saw : {o['message']}"]
        scan = o.get("scan")
        if scan:
            for c in scan["clues"]:
                if not c["already_known"]:
                    lines.append(f"   clue: [{c['id']}] {c['text']}")
        lines.append(f"   belief: estimate {tuple(b['estimate'])}, confidence {b['confidence']:.0%}, "
                     f"entropy {b['entropy_bits']:.2f} bits")
        return "\n".join(lines) + "\n"
