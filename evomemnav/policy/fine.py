"""Fine stage: candidate selection and goal verification (Section 3.3)."""

from typing import Optional, Sequence, Tuple

import numpy as np

from evomemnav.vlm.prompts import (
    SELECTION_PROMPT,
    VERIFICATION_PROMPT,
    AnchorPromptItem,
    FrontierPromptItem,
    Selection,
    Verification,
    build_selection_query,
    build_verification_query,
    parse_selection,
    parse_verification,
)


def select_candidate(
    vlm,
    *,
    question: str,
    goal_image: Optional[np.ndarray],
    egocentric_view: Optional[np.ndarray],
    topo_summary: Optional[str],
    anchors: Sequence[AnchorPromptItem],
    frontiers: Sequence[FrontierPromptItem],
    max_retries: int,
) -> Optional[Selection]:
    """(a_t, sigma_t) = VLM(g, C_t). Returns None if no valid answer is obtained."""
    parts = build_selection_query(
        question=question,
        goal_image=goal_image,
        egocentric_view=egocentric_view,
        topo_summary=topo_summary,
        anchors=anchors,
        frontiers=frontiers,
    )
    for _ in range(int(max_retries)):
        selection = parse_selection(vlm.chat(SELECTION_PROMPT, parts, kind="selection"), len(anchors), len(frontiers))
        if selection is not None:
            return selection
    return None


def apply_confidence_gate(selection: Selection, n_frontiers: int) -> Tuple[str, int]:
    """An anchor view selected with a tag other than ``certain`` is replaced by
    the highest-ranked frontier view, when one exists."""
    if selection.kind == "anchor" and selection.status != "certain" and n_frontiers > 0:
        return "frontier", 0
    return selection.kind, selection.index


def verify_goal(
    vlm,
    *,
    question: str,
    task_type: str,
    target_class: str,
    visible_objects: Sequence[str],
    experience: Optional[dict],
    current_view: np.ndarray,
    neighbor_views: Sequence[np.ndarray],
    goal_image: Optional[np.ndarray],
    target_crop: Optional[np.ndarray],
    tau: float,
) -> Verification:
    """Returns d_t in {STOP, RESELECT}; a failed call counts as RESELECT."""
    parts = build_verification_query(
        question=question,
        task_type=task_type,
        target_class=target_class,
        visible_objects=visible_objects,
        experience=experience,
        current_view=current_view,
        neighbor_views=neighbor_views,
        goal_image=goal_image,
        target_crop=target_crop,
    )
    return parse_verification(vlm.chat(VERIFICATION_PROMPT, parts, kind="verification"), tau)
