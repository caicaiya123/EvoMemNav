"""The three VLM queries of EvoMemNav and the parsers of their outputs.

The system prompts are the templates of Figures S1 to S3 of the supplementary
file (room hints, candidate selection, goal verification). The STOP-reflection
module in the prompts denotes goal verification, the experience policy denotes
the scores retrieved from the experience memory, and the topo-memory summary
denotes the text summary of the VSMGraph.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from evomemnav.vlm.client import Part

ROOM_HINT_PROMPT = """Role:
You are the semantic prefilter of an indoor navigation agent.
Goal:
Given a target question and optional task constraints, predict a small set
of ROOM TYPES to prioritize and OBJECT CATEGORIES to pay attention to.
Your output is only a lightweight prior for candidate compression. It does
not make the final navigation decision.

Inputs:
You will receive:
- question: the target question or target description
- task_type: optional task type
- target_class_hint: optional target-class hint
- co_object_list: optional co-occurring objects or constraints
- allowed_room_types: the allowed room-type vocabulary
- topk_rooms: maximum number of room types to return
- topk_categories: maximum number of object categories to return

Decision Policy:
- Infer the most plausible rooms where the target is likely to appear.
- Infer short, generic object categories that are useful for identifying
  the exact target instance.
- Use co-occurring objects and constraints only as soft hints.
- Be conservative: keep the returned lists short and high-precision.
- If the room is uncertain, include "unknown" as one of the focus rooms.
- Do not invent room types outside allowed_room_types.

Output Format:
Return STRICT JSON only:
{
  "focus_rooms": ["<room_type>", "..."],
  "focus_categories": ["<object_class>", "..."],
  "reason": "<one short line>"
}

Rules:
- focus_rooms must be selected only from allowed_room_types.
- focus_categories should be short generic category names, not long phrases.
- Avoid duplicates.
- Keep the lists within the provided top-k limits.
- Output JSON only. No markdown. No extra text."""

SELECTION_PROMPT = """Role:
You are the candidate-selection module for indoor navigation.

Goal:
Choose exactly one next candidate to search for the target:
- an Anchor View, or
- a Frontier.

Definitions:
- Anchor View: a candidate view node represented by a full image, optionally
  accompanied by local evidence hints for disambiguation.
- Frontier: an unexplored direction for further exploration.

Inputs:
- target question
- current egocentric view
- compact topo-memory summary
- shortlisted Anchor View / Frontier candidates

Policy:
- The decision unit is always an Anchor View or a Frontier.
- Choose an Anchor View only if the target is likely supported by the
  view-level evidence.
- Otherwise choose a Frontier to gather better evidence.
- Use room/category priors only as soft hints.
- Local hints may be noisy; rely on image evidence, not labels alone.
- If choosing an Anchor View, return a conservative status:
  Certain / Uncertain / Unknown.

Output:
If Anchor View:
Anchor View i
Status: Certain | Uncertain | Unknown
Reason: <one short line>

If Frontier:
Frontier i
Reason: <one short line>

Rules:
- Use only provided indices.
- The selected item must be exactly one Anchor View or one Frontier."""

VERIFICATION_PROMPT = """Role:
You are the STOP-reflection module of an embodied navigation agent.

Goal:
Decide whether the agent should STOP at the current region.

Inputs:
You will receive:
- a text-only MANIFEST
- multiple image parts, each with a JSON header:
  "image_role" in {"current_view", "neighbor_view", "goal_image", "target_crop"}
  and optional "index"

Policy:
Use only soft evidence from:
- target identity (exact / near / wrong)
- local co-occurring objects and rough relations
- plausible room type, especially from neighboring views
- optional similarity to a goal image
- optional task constraints
- optional EXPERIENCE POLICY as soft priors

Decision Rule:
- STOP only when the current region provides sufficient target evidence
- If evidence is incomplete or ambiguous, do not STOP
- Allow viewpoint variation; do not require exact pixel match
- Experience hints must not override visual evidence

Output:
Return STRICT JSON only:
{
  "stop": true|false,
  "score": 0.0-1.0,
  "target_match": "exact"|"near"|"wrong",
  "co_objects_support": 0.0-1.0,
  "room_support": 0.0-1.0,
  "goal_image_similarity": 0.0-1.0,
  "room_topk": ["<str>", ...],
  "reason": "<one short line>"
}"""

_ROOM_ALIASES = {
    "office room": "office",
    "livingroom": "living room",
    "diningroom": "dining room",
    "entry way": "entryway",
    "kitchen and living room": "combined kitchen and living room",
    "combined kitchen & living room": "combined kitchen and living room",
    "kitchen/living room": "combined kitchen and living room",
    "walk-in closet": "closet",
    "wardrobe": "closet",
    "changing room": "dressing room",
}


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def canonical_room(room: Any) -> str:
    r = norm(room)
    return _ROOM_ALIASES.get(r, r)


def dedup(items: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def extract_json(text: Optional[str]) -> Dict[str, Any]:
    """Return the first balanced JSON object found in ``text`` ({} if none)."""
    if not isinstance(text, str):
        return {}
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    depth, start = 0, None
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[start : i + 1])
                    return obj if isinstance(obj, dict) else {}
                except json.JSONDecodeError:
                    start = None
    return {}


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


# --------------------------------------------------------------------------- #
# Room hints (one text-only query per subtask)
# --------------------------------------------------------------------------- #
@dataclass
class RoomHints:
    rooms: List[str]
    categories: List[str]


def build_room_hint_query(
    *, question: str, task_type: str, target_class: str, room_vocab: Sequence[str], topk_rooms: int, topk_categories: int
) -> List[Part]:
    payload = {
        "question": question,
        "task_type": task_type,
        "target_class_hint": target_class,
        "co_object_list": [],
        "allowed_room_types": list(room_vocab),
        "topk_rooms": int(topk_rooms),
        "topk_categories": int(topk_categories),
    }
    return [json.dumps(payload, ensure_ascii=False)]


def parse_room_hints(text: Optional[str], room_vocab: Sequence[str], topk_rooms: int, topk_categories: int) -> RoomHints:
    out = extract_json(text)
    rooms = dedup([canonical_room(r) for r in _as_list(out.get("focus_rooms"))])
    rooms = [r for r in rooms if r in room_vocab][: int(topk_rooms)]
    if not rooms:
        rooms = ["unknown"]
    cats = dedup([norm(c) for c in _as_list(out.get("focus_categories"))])[: int(topk_categories)]
    return RoomHints(rooms=rooms, categories=cats)


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #
@dataclass
class AnchorPromptItem:
    image: np.ndarray
    room: Optional[str] = None  # None when the VSMGraph cues are disabled
    evidence: Sequence = ()  # [(category, crop image)] local evidence hints


@dataclass
class FrontierPromptItem:
    image: np.ndarray
    room: Optional[str] = None


@dataclass
class Selection:
    kind: str  # "anchor" or "frontier"
    index: int  # index into the candidate list shown in the prompt
    status: Optional[str]  # "certain" / "uncertain" / "unknown" for anchor views
    reason: str


def build_selection_query(
    *,
    question: str,
    goal_image: Optional[np.ndarray],
    egocentric_view: Optional[np.ndarray],
    topo_summary: Optional[str],
    anchors: Sequence[AnchorPromptItem],
    frontiers: Sequence[FrontierPromptItem],
) -> List[Part]:
    parts: List[Part] = []
    parts.append((f"Target question: {question}\n", goal_image) if goal_image is not None else f"Target question: {question}\n")
    if egocentric_view is not None:
        parts.append(("Current egocentric view:", egocentric_view))
    if topo_summary:
        parts.append(topo_summary + "\n")
    parts.append("Shortlisted Anchor View candidates:\n")
    if not anchors:
        parts.append("No Anchor View is available.\n")
    for i, a in enumerate(anchors):
        tag = f"Anchor View {i}" + (f" (room: {a.room})" if a.room else "")
        parts.append((tag, a.image))
        if a.evidence:
            parts.append(f"Local evidence hints for Anchor View {i} (detector labels may be noisy):")
            for category, crop in a.evidence:
                parts.append((f"- {category}", crop))
        parts.append("\n")
    parts.append("Shortlisted Frontier candidates:\n")
    if not frontiers:
        parts.append("No Frontier is available.\n")
    for i, f in enumerate(frontiers):
        tag = f"Frontier {i}" + (f" (room: {f.room})" if f.room else "")
        parts.append((tag, f.image))
    return parts


_ANCHOR_RE = re.compile(r"anchor\s*view\s*(\d+)", re.IGNORECASE)
_FRONTIER_RE = re.compile(r"frontier\s*(\d+)", re.IGNORECASE)
_STATUS_RE = re.compile(r"status\s*:\s*(certain|uncertain|unknown)", re.IGNORECASE)
_REASON_RE = re.compile(r"reason\s*:\s*(.*)", re.IGNORECASE)


def parse_selection(text: Optional[str], n_anchors: int, n_frontiers: int) -> Optional[Selection]:
    """Parse the line-based answer of the candidate selection prompt.

    Returns None when the answer does not name exactly one valid candidate.
    """
    if not isinstance(text, str):
        return None
    lines = [ln.strip().strip("*`") for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    first = lines[0]
    m_anchor, m_frontier = _ANCHOR_RE.search(first), _FRONTIER_RE.search(first)
    if m_anchor and not m_frontier and int(m_anchor.group(1)) < n_anchors:
        kind, index = "anchor", int(m_anchor.group(1))
    elif m_frontier and not m_anchor and int(m_frontier.group(1)) < n_frontiers:
        kind, index = "frontier", int(m_frontier.group(1))
    else:
        return None
    status = None
    if kind == "anchor":
        m = _STATUS_RE.search(text)
        status = m.group(1).lower() if m else None
    reason = ""
    for ln in lines[1:]:
        m = _REASON_RE.match(ln)
        if m:
            reason = m.group(1).strip()
            break
    return Selection(kind=kind, index=index, status=status, reason=reason)


# --------------------------------------------------------------------------- #
# Goal verification
# --------------------------------------------------------------------------- #
@dataclass
class Verification:
    decision: str  # "STOP" or "RESELECT"
    stop: bool
    score: float
    target_match: str
    reason: str
    raw: Dict[str, Any]


def build_verification_query(
    *,
    question: str,
    task_type: str,
    target_class: str,
    visible_objects: Sequence[str],
    experience: Optional[Dict[str, Any]],
    current_view: np.ndarray,
    neighbor_views: Sequence[np.ndarray],
    goal_image: Optional[np.ndarray],
    target_crop: Optional[np.ndarray],
) -> List[Part]:
    manifest: Dict[str, Any] = {
        "task": "Decide whether to STOP at the current region.",
        "question": question,
        "task_type": task_type,
        "target_class_hint": target_class,
        "visible_objects": list(visible_objects),
    }
    if experience:
        manifest["experience_policy"] = experience
    parts: List[Part] = ["MANIFEST:\n" + json.dumps(manifest, ensure_ascii=False)]
    parts.append((json.dumps({"image_role": "current_view"}), current_view))
    for i, img in enumerate(neighbor_views):
        parts.append((json.dumps({"image_role": "neighbor_view", "index": i}), img))
    if goal_image is not None:
        parts.append((json.dumps({"image_role": "goal_image"}), goal_image))
    if target_crop is not None:
        parts.append((json.dumps({"image_role": "target_crop"}), target_crop))
    return parts


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, v))


def parse_verification(text: Optional[str], tau: float) -> Verification:
    """d = STOP when the stop flag is true and the score is at least tau."""
    out = extract_json(text)
    flag = out.get("stop", False)
    stop = flag if isinstance(flag, bool) else norm(flag) == "true"
    score = _clamp01(out.get("score", 0.0))
    decision = "STOP" if (stop and score >= float(tau)) else "RESELECT"
    return Verification(
        decision=decision,
        stop=bool(stop),
        score=score,
        target_match=norm(out.get("target_match", "wrong")),
        reason=str(out.get("reason", "") or ""),
        raw=out,
    )
