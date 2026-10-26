"""Self-evolving experience memory (Section 3.4 and Appendix C).

Episode-STM keeps experience scores for the views of the current episode and is
updated after every goal verification. Scene-LTM keeps experience records per
scene, indexed by a goal key and a grid-cell node key, and is updated after each
subtask (Algorithm S2). Both memories store, for each goal:
  * anchor views: acceptances n+ and rejections n- of goal verification, with
    reliability n+ / (n+ + n-);
  * frontier views: exponential moving averages of the information gain obtained
    by exploring the frontier and of its dead-end rate.
All updates use only the agent's own verification decisions and trajectory.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from evomemnav.experience.keys import GoalKey


@dataclass
class AnchorScore:
    accept: float = 0.0
    reject: float = 0.0

    @property
    def total(self) -> float:
        return self.accept + self.reject

    @property
    def reliability(self) -> Optional[float]:
        return self.accept / self.total if self.total > 0 else None


@dataclass
class FrontierScore:
    gain: float = 0.0  # EMA of the information gain, in m^2 of newly explored area
    dead_end: float = 0.0  # EMA of the dead-end indicator
    support: float = 0.0  # number of (decayed) updates


def _ema(old: float, new: float, alpha: float, first: bool) -> float:
    return float(new) if first else (1.0 - alpha) * float(old) + alpha * float(new)


class EpisodeSTM:
    """Experience of the current episode, keyed by (goal L1 key, view id)."""

    def __init__(self, ema_alpha: float):
        self.ema_alpha = float(ema_alpha)
        self.clear()

    def clear(self):
        self.anchors: Dict[Tuple[str, str], AnchorScore] = {}
        self.frontiers: Dict[Tuple[str, str], FrontierScore] = {}
        self.suppressed: Dict[Tuple[str, str], int] = {}

    def record_verification(self, goal_l1: str, view_id: str, accepted: bool):
        s = self.anchors.setdefault((goal_l1, view_id), AnchorScore())
        if accepted:
            s.accept += 1
        else:
            s.reject += 1

    def record_frontier(self, goal_l1: str, view_id: str, gain: float, dead_end: bool):
        s = self.frontiers.setdefault((goal_l1, view_id), FrontierScore())
        first = s.support == 0
        s.gain = _ema(s.gain, gain, self.ema_alpha, first)
        s.dead_end = _ema(s.dead_end, float(dead_end), self.ema_alpha, first)
        s.support += 1

    def suppress(self, goal_l1: str, view_id: str, steps: int):
        key = (goal_l1, view_id)
        self.suppressed[key] = max(int(steps), self.suppressed.get(key, 0))

    def is_suppressed(self, goal_l1: str, view_id: str) -> bool:
        return self.suppressed.get((goal_l1, view_id), 0) > 0

    def tick(self):
        """Advance the suppression timers by one step."""
        self.suppressed = {k: v - 1 for k, v in self.suppressed.items() if v > 1}


class SceneLTM:
    """Experience records of one scene, persisted as JSON.

    ``data[kind][level_key][cell]`` holds an AnchorScore (kind ``anchors``) or a
    FrontierScore (kind ``frontiers``) as a dict.
    """

    def __init__(
        self,
        path: str,
        *,
        ema_alpha: float,
        decay: float,
        min_support: float,
        contradiction_count: float,
        max_records_per_goal: int,
        fallback_min_records: int,
    ):
        self.path = path
        self.ema_alpha = float(ema_alpha)
        self.decay = float(decay)
        self.min_support = float(min_support)
        self.contradiction_count = float(contradiction_count)
        self.max_records_per_goal = int(max_records_per_goal)
        self.fallback_min_records = int(fallback_min_records)
        self.data: Dict[str, Dict[str, Dict[str, dict]]] = {"anchors": {}, "frontiers": {}}

    # ------------------------------------------------------------------ io
    def load(self):
        self.data = {"anchors": {}, "frontiers": {}}
        if self.path and os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.data["anchors"] = loaded.get("anchors", {})
            self.data["frontiers"] = loaded.get("frontiers", {})
        return self

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------ retrieval
    def _level(self, kind: str, goal: GoalKey) -> Optional[str]:
        """Most specific goal-key level with enough records (L1, then L2, then L3)."""
        for level in goal.levels():
            if len(self.data[kind].get(level, {})) >= self.fallback_min_records:
                return level
        return None

    def anchor(self, goal: GoalKey, cell: str) -> Optional[AnchorScore]:
        level = self._level("anchors", goal)
        rec = self.data["anchors"].get(level, {}).get(cell) if level else None
        return AnchorScore(**rec) if rec else None

    def frontier(self, goal: GoalKey, cell: str) -> Optional[FrontierScore]:
        level = self._level("frontiers", goal)
        rec = self.data["frontiers"].get(level, {}).get(cell) if level else None
        return FrontierScore(**rec) if rec else None

    # --------------------------------------------------------------- update
    def update(
        self,
        goal: GoalKey,
        anchor_events: List[Tuple[str, bool]],
        frontier_events: List[Tuple[str, float, bool]],
    ):
        """Algorithm S2: write the decisions and trajectory of one subtask."""
        self._decay()
        # one moving-average step per frontier cell and subtask
        per_cell: Dict[str, List[Tuple[float, bool]]] = {}
        for cell, gain, dead_end in frontier_events:
            per_cell.setdefault(cell, []).append((float(gain), bool(dead_end)))
        for level in goal.levels():
            fr = self.data["frontiers"].setdefault(level, {})
            for cell, outcomes in per_cell.items():
                s = FrontierScore(**fr[cell]) if cell in fr else FrontierScore()
                gain = sum(g for g, _ in outcomes) / len(outcomes)
                dead = sum(float(d) for _, d in outcomes) / len(outcomes)
                first = s.support == 0
                s.gain = _ema(s.gain, gain, self.ema_alpha, first)
                s.dead_end = _ema(s.dead_end, dead, self.ema_alpha, first)
                s.support += 1
                fr[cell] = asdict(s)
            an = self.data["anchors"].setdefault(level, {})
            for cell, accepted in anchor_events:
                s = AnchorScore(**an[cell]) if cell in an else AnchorScore()
                if accepted:
                    s.accept += 1
                else:
                    s.reject += 1
                an[cell] = asdict(s)
        self._prune()

    def _decay(self):
        for level in self.data["anchors"].values():
            for rec in level.values():
                rec["accept"] *= self.decay
                rec["reject"] *= self.decay
        for level in self.data["frontiers"].values():
            for rec in level.values():
                rec["support"] *= self.decay

    def _prune(self):
        """Remove contradicted or weakly supported records and bound each goal key."""
        for kind, support in (
            ("anchors", lambda r: r["accept"] + r["reject"]),
            ("frontiers", lambda r: r["support"]),
        ):
            for level_key in list(self.data[kind].keys()):
                recs = self.data[kind][level_key]
                for cell in list(recs.keys()):
                    r = recs[cell]
                    contradicted = kind == "anchors" and min(r["accept"], r["reject"]) >= self.contradiction_count
                    if contradicted or support(r) < self.min_support:
                        del recs[cell]
                if len(recs) > self.max_records_per_goal:
                    keep = sorted(recs, key=lambda c: -support(recs[c]))[: self.max_records_per_goal]
                    self.data[kind][level_key] = {c: recs[c] for c in keep}
                if not self.data[kind][level_key]:
                    del self.data[kind][level_key]

    def num_records(self) -> int:
        return sum(len(v) for kind in self.data.values() for v in kind.values())


class ExperienceMemory:
    """Episode-STM and Scene-LTM of one scene, with the update and retrieval rules."""

    def __init__(self, cfg, ltm_path: Optional[str]):
        self.suppression_steps = int(cfg.suppression_steps)
        self.stm = EpisodeSTM(cfg.ema_alpha)
        self.ltm = SceneLTM(
            ltm_path,
            ema_alpha=cfg.ema_alpha,
            decay=cfg.ltm.decay,
            min_support=cfg.ltm.min_support,
            contradiction_count=cfg.ltm.contradiction_count,
            max_records_per_goal=cfg.ltm.max_records_per_goal,
            fallback_min_records=cfg.ltm.fallback_min_records,
        )
        self._anchor_events: List[Tuple[str, bool]] = []
        self._frontier_events: List[Tuple[str, float, bool]] = []

    # ------------------------------------------------------------ lifecycle
    def start_episode(self):
        """Scene-LTM is loaded at the start of each episode of the scene."""
        self.stm.clear()
        self.ltm.load()

    def end_episode(self):
        self.stm.clear()

    def end_subtask(self, goal: GoalKey):
        """Write the subtask to Scene-LTM (Algorithm S2) and save it."""
        self.ltm.update(goal, self._anchor_events, self._frontier_events)
        self.ltm.save()
        self._anchor_events, self._frontier_events = [], []

    def tick(self):
        self.stm.tick()

    # ------------------------------------------------------------ retrieval
    def anchor_score(self, goal: GoalKey, view_id: str, cell: str) -> AnchorScore:
        """Counts of the Episode-STM entry of the view plus the Scene-LTM record of its cell."""
        total = AnchorScore()
        for s in (self.stm.anchors.get((goal.l1, view_id)), self.ltm.anchor(goal, cell)):
            if s is not None:
                total.accept += s.accept
                total.reject += s.reject
        return total

    def frontier_score(self, goal: GoalKey, view_id: str, cell: str) -> Optional[FrontierScore]:
        """Average of the Episode-STM entry and the Scene-LTM record, when present."""
        sources = [s for s in (self.stm.frontiers.get((goal.l1, view_id)), self.ltm.frontier(goal, cell)) if s is not None]
        if not sources:
            return None
        n = len(sources)
        return FrontierScore(
            gain=sum(s.gain for s in sources) / n,
            dead_end=sum(s.dead_end for s in sources) / n,
            support=sum(s.support for s in sources),
        )

    def is_suppressed(self, goal: GoalKey, view_id: str) -> bool:
        return self.stm.is_suppressed(goal.l1, view_id)

    # --------------------------------------------------------------- update
    def on_verification(self, goal: GoalKey, view_id: str, cell: str, decision: str):
        """Called after every goal verification. A rejected anchor view is
        suppressed for a fixed number of steps."""
        accepted = decision == "STOP"
        self.stm.record_verification(goal.l1, view_id, accepted)
        if not accepted:
            self.stm.suppress(goal.l1, view_id, self.suppression_steps)
        self._anchor_events.append((cell, accepted))

    def on_frontier_explored(self, goal: GoalKey, view_id: str, cell: str, gain: float, dead_end: bool):
        """Called when the agent finishes a navigation leg toward a frontier view."""
        self.stm.record_frontier(goal.l1, view_id, gain, dead_end)
        self._frontier_events.append((cell, float(gain), bool(dead_end)))
