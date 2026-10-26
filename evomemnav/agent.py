"""EvoMemNav agent: one episode of multi-modal lifelong navigation.

Each subtask follows Algorithm S1 of the supplementary file:

    obtain room hints for g with a text-only VLM query (optional)
    while the subtask is not finished:
        update the VSMGraph with the observation
        retrieve the experience scores for g and the nodes of the graph
        C_t <- filter and rank at most K_A anchor views and K_F frontier views
        if C_t^A is empty: navigate to the highest-ranked frontier view
        else:
            (a_t, sigma_t) <- VLM(g, C_t)
            if a_t is an anchor view and sigma_t != certain: a_t <- top frontier view
            navigate to a_t
            if a_t is an anchor view:
                d_t <- goal verification; update Episode-STM
                STOP -> end the subtask; RESELECT -> suppress a_t (and fall back
                to frontier views when the rejections reach the limit)
    update Scene-LTM with the decisions and trajectory of the subtask

Ground truth (goal viewpoints) is used only after the subtask, to compute SR/SPL.
"""

import logging
import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from evomemnav.env.goatbench import GoalEval, GoalSpec, build_goal, geodesic_to_viewpoints, spl
from evomemnav.env.habitat_utils import pos_habitat_to_normal, pose_habitat_to_tsdf, resize_image, start_pose
from evomemnav.experience.keys import GoalKey, cell_key, goal_content_digest
from evomemnav.experience.memory import ExperienceMemory
from evomemnav.mapping.geom import get_cam_intr, get_scene_bnds
from evomemnav.mapping.planner import TSDFPlanner
from evomemnav.memory.views import ViewRecord
from evomemnav.memory.vsmgraph import RoomLabeler, VSMGraph
from evomemnav.policy.coarse import (
    AnchorCandidate,
    CandidateSet,
    FrontierCandidate,
    category_match,
    select_candidates,
)
from evomemnav.policy.fine import apply_confidence_gate, select_candidate, verify_goal
from evomemnav.vlm.prompts import (
    ROOM_HINT_PROMPT,
    AnchorPromptItem,
    FrontierPromptItem,
    RoomHints,
    build_room_hint_query,
    parse_room_hints,
)


def unexplored_area_around(unexplored: np.ndarray, region: np.ndarray, dilate_px: int = 2) -> int:
    """Number of unexplored cells next to a frontier region."""
    from scipy import ndimage

    kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=bool)
    return int((unexplored.astype(bool) & ndimage.binary_dilation(region.astype(bool), structure=kernel)).sum())


@dataclass
class Choice:
    kind: str  # "anchor" or "frontier"
    view_id: str
    cell: str
    target: object  # ViewRecord or Frontier given to the planner
    target_obj_id: Optional[int] = None


@dataclass
class SubtaskState:
    rejections: int = 0  # rejections since the last fallback
    fallback_steps: int = 0  # remaining steps of the frontier-only fallback
    choice: Optional[Choice] = None
    frontier_leg: Optional[Dict] = None  # {"view_id", "cell", "explored0"}
    n_verifications: int = 0
    n_rejections: int = 0
    times: Dict[str, float] = field(default_factory=lambda: {"observation": 0.0, "graph": 0.0, "tsdf_frontier": 0.0, "policy": 0.0})


class EvoMemNav:
    def __init__(self, cfg, cg_cfg, vlm):
        import open_clip
        import torch
        from ultralytics import SAM, YOLOWorld

        self.cfg = cfg
        self.cg_cfg = cg_cfg
        self.vlm = vlm
        self.detection_model = YOLOWorld(cfg.perception.yolo_model)
        self.sam_predictor = SAM(cfg.perception.sam_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            cfg.perception.clip_model, cfg.perception.clip_pretrained
        )
        self.clip_model = clip_model.to(device).eval()
        self.clip_tokenizer = open_clip.get_tokenizer(cfg.perception.clip_model)

        self.room_vocab = [str(r) for r in cfg.vsmgraph.room_vocab]
        labeler = RoomLabeler(
            self.clip_model, self.clip_preprocess, self.clip_tokenizer, self.room_vocab, cfg.observation.vlm_image_size
        )
        self.graph = VSMGraph(labeler)
        self.cam_intr = get_cam_intr(cfg.camera.hfov_deg, cfg.camera.img_height, cfg.camera.img_width)
        self.global_step = 0

    # ------------------------------------------------------------------ #
    # Episode
    # ------------------------------------------------------------------ #
    def run_episode(self, scene, episode: dict, goal_types: List[str], goals: List[list], experience: ExperienceMemory, episode_tag: str):
        """Run all subtasks of one episode; yields one result record per subtask."""
        cfg = self.cfg
        pts, angle = start_pose(episode["start_position"], episode["start_rotation"])
        floor_height = pts[1]
        tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
        num_step = max(int(math.sqrt(scene_size) * cfg.planner.max_step_room_size_ratio), int(cfg.planner.min_steps))
        planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.mapping.tsdf_grid_size,
            floor_height=floor_height,
            floor_height_offset=0,
            pts_init=pts,
            init_clearance=cfg.mapping.init_clearance * 2,
            save_visualization=False,
        )
        self.graph.reset()
        experience.start_episode()

        for subtask_idx, (goal_type, subtask_goal) in enumerate(zip(goal_types, goals)):
            goal, goal_eval = build_goal(goal_type, subtask_goal, scene)
            goal_eval.shortest_path_length = geodesic_to_viewpoints(scene.pathfinder, pts, goal_eval.viewpoints) + 1e-6
            goal_key = GoalKey(
                goal.modality,
                goal.category,
                goal_content_digest(goal.modality, goal.description, goal.image.tobytes() if goal.image is not None else None),
            )
            logging.info(f"[{episode_tag}] subtask {subtask_idx + 1}/{len(goals)} {goal.modality}: {goal.question}")
            self.vlm.drain()
            t0 = perf_counter()
            pts, angle, record = self._run_subtask(scene, planner, goal, goal_eval, goal_key, pts, angle, num_step, experience)
            experience.end_subtask(goal_key)
            calls = self.vlm.drain()
            record.update(
                subtask_idx=subtask_idx,
                modality=goal.modality,
                wall_time_s=perf_counter() - t0,
                vlm_calls=len(calls),
                vlm_tokens=sum(int(c.get("total_tokens") or 0) for c in calls),
                vlm_time_s=sum(float(c.get("latency_s") or 0.0) for c in calls),
                vlm_calls_by_kind={k: sum(1 for c in calls if c["kind"] == k) for k in {c["kind"] for c in calls}},
            )
            # policy time without the VLM calls (the "EvoMem policy" term of the runtime breakdown)
            record["policy_overhead_s"] = max(0.0, record["policy_time_s"] - record["vlm_time_s"])
            logging.info(
                f"[{episode_tag}] subtask {subtask_idx + 1} success={record['success']} spl={record['spl']:.3f} "
                f"stop={record['stop_reason']} steps={record['steps']} calls={record['vlm_calls']}"
            )
            yield record

        experience.end_episode()
        self.graph.reset()

    # ------------------------------------------------------------------ #
    # Subtask
    # ------------------------------------------------------------------ #
    def _run_subtask(self, scene, planner, goal: GoalSpec, goal_eval: GoalEval, goal_key: GoalKey, pts, angle, num_step, experience):
        cfg = self.cfg
        st = SubtaskState()
        hints = self._room_hints(goal)
        planner.max_point = None
        planner.target_point = None
        path_length, stopped, stop_reason, steps = 0.0, False, "max_steps", 0
        explored = 0.0

        for cnt_step in range(num_step):
            steps = cnt_step + 1
            ego_view, main_key = self._observe(scene, planner, pts, angle, cnt_step, st)

            t0 = perf_counter()
            planner.update_frontier_map(
                pts=pts,
                cfg=cfg.planner,
                scene=scene,
                cnt_step=self.global_step,
                save_frontier_image=False,
                eps_frontier_dir=None,
                prompt_img_size=(cfg.observation.vlm_image_size, cfg.observation.vlm_image_size),
            )
            if planner.unexplored is not None:
                explored = float((planner.unexplored == 0).sum()) * cfg.mapping.tsdf_grid_size**2
            st.times["tsdf_frontier"] += perf_counter() - t0

            t0 = perf_counter()
            self.graph.add_visited_view(self.global_step, pts, main_key)
            self.graph.sync_anchor_views(scene, planner.voxel2habitat)
            frontier_nodes = self.graph.sync_frontier_views(planner.frontiers, planner.voxel2habitat)
            st.times["graph"] += perf_counter() - t0

            if planner.max_point is None and planner.target_point is None:
                t0 = perf_counter()
                self._close_frontier_leg(st, goal_key, explored, experience)
                choice = self._decide(scene, planner, goal, goal_key, hints, frontier_nodes, ego_view, pts, st, experience)
                st.times["policy"] += perf_counter() - t0
                if choice is None:
                    stop_reason = "no_valid_choice"
                    break
                if not self._set_target(planner, scene, pts, choice, frontier_nodes, st):
                    stop_reason = "navigation_point_failed"
                    break
                if choice.kind == "frontier":
                    st.frontier_leg = {"view_id": choice.view_id, "cell": choice.cell, "explored0": explored}

            rv = planner.agent_step(
                pts=pts,
                angle=angle,
                objects=scene.objects,
                anchor_views=scene.anchor_views,
                pathfinder=scene.pathfinder,
                cfg=cfg.planner,
                save_visualization=False,
            )
            if rv[0] is None:
                stop_reason = "agent_step_failed"
                break
            new_pts, angle, _, _, _, arrived = rv
            path_length += float(np.linalg.norm((np.asarray(new_pts) - np.asarray(pts))[[0, 2]]))
            pts = new_pts
            self.global_step += 1
            experience.tick()
            st.fallback_steps = max(0, st.fallback_steps - 1)

            if arrived and st.choice is not None and st.choice.kind == "anchor":
                t0 = perf_counter()
                decision = self._verify(scene, goal, goal_key, pts, angle, st.choice, experience)
                st.times["policy"] += perf_counter() - t0
                st.n_verifications += 1
                experience.on_verification(goal_key, st.choice.view_id, st.choice.cell, decision)
                if decision == "STOP":
                    stopped, stop_reason = True, "verified_stop"
                    break
                st.n_rejections += 1
                st.rejections += 1
                if st.rejections >= int(cfg.fine.max_rejections):
                    st.fallback_steps = int(cfg.fine.fallback_steps)
                    st.rejections = 0
                    logging.info(f"[FALLBACK] frontier-only for {st.fallback_steps} steps")
                st.choice = None

        self._close_frontier_leg(st, goal_key, explored, experience)

        # evaluation only: distance to the goal viewpoints
        distance = geodesic_to_viewpoints(scene.pathfinder, pts, goal_eval.viewpoints)
        success = bool(stopped and distance < float(cfg.evaluation.success_distance))
        record = {
            "success": success,
            "spl": spl(success, goal_eval.shortest_path_length, path_length),
            "stopped": stopped,
            "stop_reason": stop_reason,
            "distance_to_goal": distance if np.isfinite(distance) else None,
            "path_length": path_length,
            "shortest_path_length": goal_eval.shortest_path_length if np.isfinite(goal_eval.shortest_path_length) else None,
            "steps": steps,
            "verifications": st.n_verifications,
            "rejections": st.n_rejections,
            **{f"{k}_time_s": v for k, v in st.times.items()},
        }
        return pts, angle, record

    # ------------------------------------------------------------------ #
    # Observation and memory update
    # ------------------------------------------------------------------ #
    def _observe(self, scene, planner, pts, angle, cnt_step, st) -> Tuple[np.ndarray, str]:
        """Observe several views around the agent, update the object instances,
        the anchor views and the TSDF. Returns the forward view (resized for the
        VLM) and its image key."""
        cfg = self.cfg.observation
        if cnt_step == 0:
            n_extra, inc = cfg.first_step_extra_views, cfg.first_step_view_angle_deg * np.pi / 180
        else:
            n_extra, inc = cfg.extra_views, cfg.view_angle_deg * np.pi / 180
        total = 1 + int(n_extra)
        angles = [angle + inc * (i - total // 2) for i in range(total)]
        main_angle = angles.pop(total // 2)
        angles.append(main_angle)  # the forward view is observed last

        added: List[int] = []
        ego_view = None
        for view_idx, ang in enumerate(angles):
            t0 = perf_counter()
            obs, cam_pose = scene.get_observation(pts, angle=ang)
            rgb, depth = obs["color_sensor"], obs["depth_sensor"]
            st.times["observation"] += perf_counter() - t0

            key = f"{self.global_step}-view_{view_idx}.png"
            t0 = perf_counter()
            _, new_ids = scene.update_objects(
                image_rgb=rgb[..., :3],
                depth=depth,
                intrinsics=self.cam_intr,
                cam_pos=cam_pose,
                pts=pts,
                pts_voxel=planner.habitat2voxel(pts),
                img_path=key,
            )
            scene.observations[key] = rgb
            added += new_ids
            scene.periodic_cleanup_objects(frame_idx=cnt_step * total + view_idx, pts=pts)
            st.times["graph"] += perf_counter() - t0

            t0 = perf_counter()
            planner.integrate(
                color_im=rgb,
                depth_im=depth,
                cam_intr=self.cam_intr,
                cam_pose=pose_habitat_to_tsdf(cam_pose),
                obs_weight=1.0,
                margin_h=int(self.cfg.mapping.margin_h_ratio * self.cfg.camera.img_height),
                margin_w=int(self.cfg.mapping.margin_w_ratio * self.cfg.camera.img_width),
                explored_depth=self.cfg.mapping.explored_depth,
            )
            st.times["tsdf_frontier"] += perf_counter() - t0
            ego_view = resize_image(rgb, cfg.vlm_image_size, cfg.vlm_image_size)

        # re-cluster the new instances and the instances near the agent into anchor views
        t0 = perf_counter()
        near = self.cfg.perception.object_visibility_dist_m + 0.5
        ids = {oid for oid in added if oid in scene.objects}
        ids |= {oid for oid, obj in scene.objects.items() if np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]]) < near}
        scene.update_anchor_views(obj_ids=ids, min_detection=self.cfg.perception.min_detection)
        st.times["graph"] += perf_counter() - t0
        return ego_view, key

    # ------------------------------------------------------------------ #
    # Coarse stage
    # ------------------------------------------------------------------ #
    def _room_hints(self, goal: GoalSpec) -> RoomHints:
        cfg = self.cfg
        if not cfg.coarse.room_hints.enable:
            return RoomHints(rooms=[], categories=[])
        parts = build_room_hint_query(
            question=goal.question,
            task_type=goal.modality,
            target_class=goal.category,
            room_vocab=self.room_vocab,
            topk_rooms=cfg.coarse.room_hints.topk_rooms,
            topk_categories=cfg.coarse.room_hints.topk_categories,
        )
        text = self.vlm.chat(ROOM_HINT_PROMPT, parts, kind="room_hint")
        return parse_room_hints(text, self.room_vocab, cfg.coarse.room_hints.topk_rooms, cfg.coarse.room_hints.topk_categories)

    def _cell(self, position) -> str:
        return cell_key((position[0], position[2]), self.cfg.experience.ltm.cell_size_m)

    def _candidates(self, scene, planner, goal, goal_key, hints, frontier_nodes, pts, st, experience) -> CandidateSet:
        cfg = self.cfg
        anchors = []
        for recency, (key, node) in enumerate(self.graph.anchors.items()):
            reliability = experience.anchor_score(goal_key, key, self._cell(node.position)).reliability
            anchors.append(
                AnchorCandidate(
                    view_id=key,
                    room=node.room,
                    categories=[scene.objects[o]["class_name"] for o in node.objects if o in scene.objects],
                    recency=recency,
                    reliability=reliability,
                    suppressed=experience.is_suppressed(goal_key, key),
                )
            )

        cur_voxel = planner.habitat2voxel(pts)[:2]
        frontiers = []
        for i, (ft, node) in enumerate(zip(planner.frontiers, frontier_nodes)):
            score = experience.frontier_score(goal_key, node.id, self._cell(node.position))
            dist, _ = planner.get_distance(cur_voxel, ft.position[:2], pos_habitat_to_normal(pts)[2], scene.pathfinder)
            if dist is None or not np.isfinite(dist) or dist <= 0:
                dist = float(np.linalg.norm(ft.position[:2] - cur_voxel) * cfg.mapping.tsdf_grid_size)
            frontiers.append(
                FrontierCandidate(
                    index=i,
                    view_id=node.id,
                    room=node.room,
                    unexplored_area=unexplored_area_around(planner.unexplored, ft.region),
                    path_distance=float(dist),
                    gain=score.gain if score is not None else None,
                    dead_end=score.dead_end if score is not None else None,
                )
            )

        return select_candidates(
            anchors,
            frontiers,
            target=goal.category,
            room_hints=hints.rooms,
            category_hints=hints.categories,
            k_a=cfg.coarse.anchor_budget,
            k_f=cfg.coarse.frontier_budget,
            w_gain=cfg.coarse.frontier_weights.gain,
            w_distance=cfg.coarse.frontier_weights.distance,
            frontier_only=st.fallback_steps > 0 and len(frontiers) > 0,
        )

    # ------------------------------------------------------------------ #
    # Fine stage
    # ------------------------------------------------------------------ #
    def _decide(self, scene, planner, goal, goal_key, hints, frontier_nodes, ego_view, pts, st, experience) -> Optional[Choice]:
        cfg = self.cfg
        cands = self._candidates(scene, planner, goal, goal_key, hints, frontier_nodes, pts, st, experience)
        if not cands.anchors:
            if not cands.frontiers:
                return None
            return self._frontier_choice(planner, cands.frontiers[0])

        size = int(cfg.observation.vlm_image_size)
        anchor_items = [
            AnchorPromptItem(
                image=resize_image(scene.observations[a.view_id], size, size),
                room=a.room,
                evidence=self._evidence_hints(scene, a.view_id, goal.category, hints.categories),
            )
            for a in cands.anchors
        ]
        frontier_items = [FrontierPromptItem(image=planner.frontiers[f.index].feature, room=f.room) for f in cands.frontiers]
        selection = select_candidate(
            self.vlm,
            question=goal.question,
            goal_image=resize_image(goal.image, size, size) if goal.image is not None else None,
            egocentric_view=ego_view if cfg.selection.include_egocentric_view else None,
            topo_summary=self.graph.summary(hints.rooms, hints.categories),
            anchors=anchor_items,
            frontiers=frontier_items,
            max_retries=cfg.selection.max_retries,
        )
        if selection is None:
            return None
        kind, index = apply_confidence_gate(selection, len(cands.frontiers))
        logging.info(f"[SELECT] {selection.kind} {selection.index} status={selection.status} -> {kind} {index}: {selection.reason}")
        if kind == "frontier":
            return self._frontier_choice(planner, cands.frontiers[index])
        return self._anchor_choice(scene, cands.anchors[index].view_id, goal.category)

    def _frontier_choice(self, planner, cand: FrontierCandidate) -> Choice:
        node = self.graph.frontiers[cand.view_id]
        return Choice("frontier", cand.view_id, self._cell(node.position), planner.frontiers[cand.index])

    def _anchor_choice(self, scene, view_id: str, target_category: str) -> Choice:
        """Navigate to observe the target instance of the anchor view (the most
        confident visible object of the target category), or the instances
        assigned to the view when no such object is visible."""
        view = scene.anchor_views[view_id]
        hits = [
            o for o in scene.visible_objects(view_id) if category_match(scene.objects[o]["class_name"], [target_category.lower()])
        ]
        target_obj = max(hits, key=lambda o: float(scene.objects[o]["conf"])) if hits else None
        cluster = [target_obj] if target_obj is not None else list(view.cluster)
        target = ViewRecord(
            image=view.image,
            color=view.color,
            obs_point=view.obs_point,
            full_obj_list={o: float(scene.objects[o]["conf"]) for o in cluster},
            cluster=cluster,
        )
        node = self.graph.anchors[view_id]
        return Choice("anchor", view_id, self._cell(node.position), target, target_obj)

    def _evidence_hints(self, scene, view_id, target_category, category_hints) -> List[Tuple[str, np.ndarray]]:
        """Local evidence hints of an anchor view: crops of its visible objects,
        prioritizing the target category and the hinted categories."""
        cfg = self.cfg.selection
        size = int(self.cfg.observation.vlm_image_size) // 3
        objs = scene.visible_objects(view_id)
        terms_t, terms_h = [target_category.lower()], [c.lower() for c in category_hints]

        def priority(o):
            name = scene.objects[o]["class_name"]
            return (
                0 if category_match(name, terms_t) else (1 if category_match(name, terms_h) else 2),
                -float(scene.objects[o]["conf"]),
            )

        out, per_class = [], {}
        for o in sorted(objs, key=priority):
            name = scene.objects[o]["class_name"]
            if per_class.get(name, 0) >= int(cfg.max_crops_per_class):
                continue
            crop = scene.object_crop(view_id, o, size=(size, size))
            if crop is None:
                continue
            out.append((name, crop))
            per_class[name] = per_class.get(name, 0) + 1
            if len(out) >= int(cfg.max_crops_per_anchor):
                break
        return out

    def _set_target(self, planner, scene, pts, choice: Choice, frontier_nodes, st) -> bool:
        ok = planner.set_next_navigation_point(
            choice=choice.target, pts=pts, objects=scene.objects, cfg=self.cfg.planner, pathfinder=scene.pathfinder
        )
        if ok and choice.kind == "frontier":
            # a frontier whose navigation point is the current cell would not move the agent
            cur = planner.habitat2voxel(pts)[:2].astype(int)
            if planner.target_point is not None and np.array_equal(np.asarray(planner.target_point).astype(int)[:2], cur):
                for ft, node in zip(planner.frontiers, frontier_nodes):
                    if ft is choice.target:
                        continue
                    planner.max_point, planner.target_point = None, None
                    if planner.set_next_navigation_point(
                        choice=ft, pts=pts, objects=scene.objects, cfg=self.cfg.planner, pathfinder=scene.pathfinder
                    ) and not np.array_equal(np.asarray(planner.target_point).astype(int)[:2], cur):
                        choice = Choice("frontier", node.id, self._cell(node.position), ft)
                        break
        st.choice = choice if ok else None
        return ok

    def _close_frontier_leg(self, st, goal_key, explored, experience):
        """Information gain of a finished frontier leg; a leg with little gain is a dead end."""
        leg = st.frontier_leg
        if leg is None:
            return
        gain = max(0.0, explored - leg["explored0"])
        experience.on_frontier_explored(
            goal_key, leg["view_id"], leg["cell"], gain, gain < float(self.cfg.experience.dead_end_gain_m2)
        )
        st.frontier_leg = None

    def _verify(self, scene, goal: GoalSpec, goal_key: GoalKey, pts, angle, choice: Choice, experience) -> str:
        cfg = self.cfg
        size = int(cfg.observation.vlm_image_size)
        obs, _ = scene.get_observation(pts, angle=angle)
        current = resize_image(obs["color_sensor"], size, size)
        neighbors = []
        delta = cfg.observation.neighbor_view_angle_deg * np.pi / 180
        for sign in (+1, -1):
            obs_n, _ = scene.get_observation(pts, angle=angle + sign * delta)
            neighbors.append(resize_image(obs_n["color_sensor"], size, size))

        visible = []
        for o in scene.visible_objects(choice.view_id):
            name = scene.objects[o]["class_name"]
            if name not in visible:
                visible.append(name)

        experience_hint = None
        score = experience.anchor_score(goal_key, choice.view_id, choice.cell)
        if score.total > 0:
            experience_hint = {
                "acceptances": round(score.accept, 2),
                "rejections": round(score.reject, 2),
                "reliability": round(score.reliability, 2),
            }

        crop = None
        if choice.target_obj_id is not None and choice.target_obj_id in scene.objects:
            crop = scene.object_crop(choice.view_id, choice.target_obj_id, size=(size, size))

        result = verify_goal(
            self.vlm,
            question=goal.question,
            task_type=goal.modality,
            target_class=goal.category,
            visible_objects=visible,
            experience=experience_hint,
            current_view=current,
            neighbor_views=neighbors,
            goal_image=resize_image(goal.image, size, size) if goal.image is not None else None,
            target_crop=crop,
            tau=cfg.fine.verify_threshold,
        )
        logging.info(f"[VERIFY] {choice.view_id} -> {result.decision} (stop={result.stop}, score={result.score:.2f}): {result.reason}")
        return result.decision
