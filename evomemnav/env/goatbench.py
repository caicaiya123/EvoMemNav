"""GOAT-Bench episodes, goals and metrics.

The agent receives only ``GoalSpec`` (modality, target category, question,
description and goal image). ``GoalEval`` holds the goal viewpoints and the
shortest-path length, which are used only to compute SR and SPL.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class GoalSpec:
    modality: str  # "object", "description" or "image"
    category: str
    question: str
    description: Optional[str] = None
    image: Optional[np.ndarray] = None  # goal image for image goals


@dataclass
class GoalEval:
    viewpoints: List[List[float]] = field(default_factory=list)
    shortest_path_length: float = float("inf")


def list_scene_files(goatbench_dir: str) -> List[str]:
    return sorted(f for f in os.listdir(goatbench_dir) if f.endswith(".json") and not f.startswith("."))


def load_scene_file(goatbench_dir: str, scene_file: str) -> dict:
    with open(os.path.join(goatbench_dir, scene_file), "r") as f:
        return json.load(f)


def resolve_scene_id(scene_name: str, all_scene_ids: List[str]) -> str:
    """Map a GOAT-Bench scene name (e.g. 4ok3usBNeis) to its HM3D folder (e.g. 00824-4ok3usBNeis)."""
    for sid in all_scene_ids:
        if sid == scene_name or sid.endswith(f"-{scene_name}"):
            return sid
    raise FileNotFoundError(f"Scene {scene_name} not found in the HM3D scene folders")


def prepare_goatbench_navigation_goals(scene_name: str, episode: dict, all_navigation_goals: dict):
    """Expand the subtasks of an episode into goal entries (following GOAT-Bench).

    Language goals with descriptions longer than 55 words are skipped, as in the
    GOAT-Bench evaluation code.
    """
    filtered_tasks = []
    for goal in episode["tasks"]:
        goal_category, goal_type, goal_inst_id = goal[0], goal[1], goal[2]
        same_cat = [x for x in all_navigation_goals.values() if x[0]["object_category"] == goal_category]
        if goal_type == "description":
            inst = [x for x in same_cat[0] if x["object_id"] == goal_inst_id]
            if len(inst[0]["lang_desc"].split(" ")) <= 55:
                filtered_tasks.append(goal)
        else:
            filtered_tasks.append(goal)

    goal_types, goals = [], []
    for goal in filtered_tasks:
        goal_category, goal_type, goal_inst_id = goal[0], goal[1], goal[2]
        goal_types.append(goal_type)
        same_cat = [x for x in all_navigation_goals.values() if x[0]["object_category"] == goal_category]
        for child_category in same_cat[0][0]["children_object_categories"]:
            goal_key = f"{scene_name}.basis.glb_{child_category}"
            if goal_key in all_navigation_goals:
                same_cat[0].extend(all_navigation_goals[goal_key])
        assert len(same_cat) == 1, f"more than 1 goal categories for {goal_category}"
        if goal_type == "object":
            goals.append(same_cat[0])
        else:
            goals.append([x for x in same_cat[0] if x["object_id"] == goal_inst_id])
    return goal_types, goals


def build_goal(goal_type: str, subtask_goal: List[dict], scene) -> Tuple[GoalSpec, GoalEval]:
    """Build the goal given to the agent and the evaluation data of a subtask.

    The image of an image goal is rendered at the goal viewpoint provided by the
    dataset, as in GOAT-Bench.
    """
    category = subtask_goal[0]["object_category"]
    description, image = None, None
    if goal_type == "object":
        question = f"Can you find the {category}?"
    elif goal_type == "description":
        description = subtask_goal[0].get("lang_desc", subtask_goal[0].get("question"))
        question = f"Could you find the object exactly described as the '{description}'?"
    else:
        question = (
            "Could you find the exact object captured at the center of the following image? "
            "You need to pay attention to the environment and find the exact object."
        )
        state = subtask_goal[0]["view_points"][0]["agent_state"]
        obs, _ = scene.get_observation(pts=state["position"], rotation=state["rotation"])
        image = obs["color_sensor"]
    viewpoints = [vp["agent_state"]["position"] for g in subtask_goal for vp in g["view_points"]]
    return GoalSpec(goal_type, category, question, description, image), GoalEval(viewpoints=viewpoints)


def geodesic_to_viewpoints(pathfinder, pts, viewpoints) -> float:
    """Geodesic distance from ``pts`` to the nearest goal viewpoint."""
    import habitat_sim

    best = np.inf
    for viewpoint in viewpoints:
        path = habitat_sim.ShortestPath()
        path.requested_start = pts
        path.requested_end = viewpoint
        if pathfinder.find_path(path):
            best = min(best, path.geodesic_distance)
    return float(best)


def spl(success: bool, shortest_path_length: float, path_length: float) -> float:
    if not success:
        return 0.0
    return float(shortest_path_length / max(shortest_path_length, path_length))
