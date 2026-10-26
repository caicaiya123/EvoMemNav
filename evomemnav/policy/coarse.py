"""Coarse stage: filter and rank at most K_A anchor views and K_F frontier views.

Ranking (Section 3.3 and Appendix B of the supplementary file):
  * views are grouped by their room labels, with priority to the rooms suggested
    by the room hints;
  * anchor views whose visible objects contain the target category, or a
    category suggested by the room hints, are ranked first;
  * frontier views are ranked by the size of their unexplored region and their
    path distance from the agent;
  * experience scores reorder the views within this ranking: anchor views with
    higher reliability and frontier views with higher gain and lower dead-end
    rate move up;
  * suppressed anchor views are removed, and the top K_A anchor views and top
    K_F frontier views form the candidate set.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class AnchorCandidate:
    view_id: str  # key of the anchor view in the VSMGraph
    room: str
    categories: List[str]  # categories of the visible objects O^vis(v)
    recency: int  # larger is more recent
    reliability: Optional[float] = None  # n+ / (n+ + n-), None without records
    suppressed: bool = False


@dataclass
class FrontierCandidate:
    index: int  # index into the planner's frontier list
    view_id: str
    room: str
    unexplored_area: float  # unexplored cells around the frontier region
    path_distance: float  # meters
    gain: Optional[float] = None  # experience: EMA of information gain (m^2)
    dead_end: Optional[float] = None  # experience: EMA of dead-end rate


@dataclass
class CandidateSet:
    anchors: List[AnchorCandidate] = field(default_factory=list)
    frontiers: List[FrontierCandidate] = field(default_factory=list)


def category_match(category: str, terms: Sequence[str]) -> bool:
    """Detector categories may carry modifiers, so a term also matches as a substring."""
    c = category.strip().lower()
    return any(t and (c == t or t in c) for t in terms)


def _minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rank_anchors(
    anchors: Sequence[AnchorCandidate],
    *,
    target: str,
    room_hints: Sequence[str],
    category_hints: Sequence[str],
) -> List[AnchorCandidate]:
    terms = [t.strip().lower() for t in [target, *category_hints] if t]
    rooms = {r for r in room_hints if r and r != "unknown"}

    def key(a: AnchorCandidate):
        hit = any(category_match(c, terms) for c in a.categories)
        in_hinted_room = a.room in rooms
        exp = (a.reliability - 0.5) if a.reliability is not None else 0.0
        return (0 if hit else 1, 0 if in_hinted_room else 1, -exp, -a.recency)

    return sorted((a for a in anchors if not a.suppressed), key=key)


def frontier_heuristic(frontiers: Sequence[FrontierCandidate], w_gain: float, w_distance: float) -> List[float]:
    gain = _minmax([f.unexplored_area for f in frontiers])
    dist = _minmax([f.path_distance for f in frontiers])
    return [w_gain * g - w_distance * d for g, d in zip(gain, dist)]


def frontier_experience(frontiers: Sequence[FrontierCandidate]) -> List[float]:
    """Normalized gain minus dead-end rate; 0 for frontier views without records."""
    gains = [f.gain for f in frontiers if f.gain is not None]
    g_max = max(gains) if gains else 0.0
    out = []
    for f in frontiers:
        if f.gain is None:
            out.append(0.0)
            continue
        g = f.gain / g_max if g_max > 1e-9 else 0.5
        out.append(g - float(f.dead_end or 0.0))
    return out


def rank_frontiers(
    frontiers: Sequence[FrontierCandidate],
    *,
    room_hints: Sequence[str],
    w_gain: float,
    w_distance: float,
) -> List[FrontierCandidate]:
    rooms = {r for r in room_hints if r and r != "unknown"}
    heur = frontier_heuristic(frontiers, w_gain, w_distance)
    exp = frontier_experience(frontiers)
    order = sorted(
        range(len(frontiers)),
        key=lambda i: (0 if frontiers[i].room in rooms else 1, -exp[i], -heur[i]),
    )
    return [frontiers[i] for i in order]


def select_candidates(
    anchors: Sequence[AnchorCandidate],
    frontiers: Sequence[FrontierCandidate],
    *,
    target: str,
    room_hints: Sequence[str],
    category_hints: Sequence[str],
    k_a: int,
    k_f: int,
    w_gain: float,
    w_distance: float,
    frontier_only: bool = False,
) -> CandidateSet:
    ranked_a = rank_anchors(anchors, target=target, room_hints=room_hints, category_hints=category_hints)
    ranked_f = rank_frontiers(frontiers, room_hints=room_hints, w_gain=w_gain, w_distance=w_distance)
    return CandidateSet(
        anchors=[] if frontier_only else ranked_a[:k_a],
        frontiers=ranked_f[:k_f],
    )
