"""Per-subtask result records and their summary (SR, SPL and cost per subtask)."""

import glob
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

MODALITIES = ("object", "description", "image")


class ResultRecorder:
    """Appends one JSON line per subtask to ``<output_dir>/subtasks_<tag>.jsonl``."""

    def __init__(self, output_dir: str, tag: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.path = os.path.join(output_dir, f"subtasks_{tag}.jsonl")

    def finished_episodes(self) -> Dict[str, int]:
        """Number of recorded subtasks per episode key, over all record files."""
        counts: Dict[str, set] = defaultdict(set)
        for rec in load_records(self.output_dir):
            counts[rec["episode_key"]].add(rec["subtask_idx"])
        return {k: len(v) for k, v in counts.items()}

    def log(self, record: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def load_records(output_dir: str) -> List[dict]:
    """All subtask records of a run; a re-run subtask keeps its latest record."""
    latest = {}
    for path in sorted(glob.glob(os.path.join(output_dir, "subtasks_*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    latest[rec["subtask_id"]] = rec
    return list(latest.values())


def _mean_std(values: Iterable[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(v.mean()), "std": float(v.std())}


def summarize(records: List[dict]) -> dict:
    """SR and SPL in percent, overall and per goal modality, and cost per subtask."""
    out = {"n_subtasks": len(records)}
    if not records:
        return out
    out["SR"] = 100.0 * float(np.mean([r["success"] for r in records]))
    out["SPL"] = 100.0 * float(np.mean([r["spl"] for r in records]))
    for m in MODALITIES:
        sub = [r for r in records if r["modality"] == m]
        if sub:
            out[m] = {
                "n": len(sub),
                "SR": 100.0 * float(np.mean([r["success"] for r in sub])),
                "SPL": 100.0 * float(np.mean([r["spl"] for r in sub])),
            }
    out["cost_per_subtask"] = {
        "vlm_calls": _mean_std(r.get("vlm_calls") for r in records),
        "vlm_tokens": _mean_std(r.get("vlm_tokens") for r in records),
        "vlm_time_s": _mean_std(r.get("vlm_time_s") for r in records),
        "graph_time_s": _mean_std(r.get("graph_time_s") for r in records),
        "steps": _mean_std(r.get("steps") for r in records),
    }
    return out
