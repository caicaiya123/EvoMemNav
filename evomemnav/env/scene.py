"""Habitat scene with the observation memory of EvoMemNav.

The scene owns the simulator and the parts of the VSMGraph that are built from
images (Appendix A of the supplementary file):
  * object instances: YOLO-World detections segmented by SAM, back-projected
    with depth and merged across frames (``objects``);
  * view records: every observed image with detections and its visible objects
    (``views``), and the anchor views selected from them by co-visibility
    clustering (``anchor_views``);
  * the raw images (``observations``), which are given to the VLM.
"""

import copy
import logging
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import habitat_sim
import numpy as np
import quaternion
import supervision as sv
import torch
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_two_vectors, quat_to_coeffs

from evomemnav.env.habitat_utils import get_quaternion, make_sim_cfg, rgba2rgb
from evomemnav.memory.view_clustering import SceneHierarchicalClustering
from evomemnav.memory.views import ViewRecord
from evomemnav.third_party.conceptgraph.slam.mapping import (
    aggregate_similarities,
    compute_spatial_similarities,
    compute_visual_similarities,
    match_detections_to_objects,
)
from evomemnav.third_party.conceptgraph.slam.slam_classes import DetectionDict, MapObjectDict, to_tensor
from evomemnav.third_party.conceptgraph.slam.utils import (
    denoise_objects,
    detections_to_obj_pcd_and_bbox,
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    merge_obj2_into_obj1,
    merge_objects,
    processing_needed,
    resize_gobs,
)
from evomemnav.third_party.conceptgraph.utils.general_utils import ObjectClasses, filter_detections, measure_time
from evomemnav.third_party.conceptgraph.utils.ious import mask_subtract_contained
from evomemnav.third_party.conceptgraph.utils.model_utils import compute_clip_features_batched


class Scene:
    def __init__(
        self,
        scene_id: str,
        cfg,
        cg_cfg,
        detection_model,
        sam_predictor,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
    ):
        self.cfg = cfg
        self.cfg_cg = cg_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
        scene_dir = os.path.join(cfg.data.scene_data_path, split, scene_id)
        mesh_path = os.path.join(scene_dir, scene_id.split("-")[1] + ".basis.glb")
        navmesh_path = os.path.join(scene_dir, scene_id.split("-")[1] + ".basis.navmesh")
        assert os.path.exists(mesh_path), f"{mesh_path} does not exist"
        assert os.path.exists(navmesh_path), f"{navmesh_path} does not exist"

        sim_settings = {
            "scene": mesh_path,
            "default_agent": 0,
            "sensor_height": cfg.camera.height_m,
            "width": cfg.camera.img_width,
            "height": cfg.camera.img_height,
            "hfov": cfg.camera.hfov_deg,
            "scene_dataset_config_file": cfg.data.scene_dataset_config_path,
            "camera_tilt": cfg.camera.tilt_deg * np.pi / 180,
            "gpu_device_id": int(os.environ.get("HABITAT_SIM_GPU_DEVICE_ID", "0")),
        }
        self.simulator = habitat_sim.Simulator(make_sim_cfg(sim_settings))
        self.pathfinder = self.simulator.pathfinder
        self.pathfinder.seed(cfg.seed)
        self.pathfinder.load_nav_mesh(navmesh_path)
        self.agent = self.simulator.initialize_agent(sim_settings["default_agent"])
        logging.info(f"Loaded scene {scene_id}")

        self.obj_classes = ObjectClasses(
            classes_file_path=cfg.data.object_classes_path,
            bg_classes=self.cfg_cg["bg_classes"],
        )

        self.objects: MapObjectDict[int, Dict] = MapObjectDict()  # object id -> instance
        self.object_id_counter = 1
        self.views: Dict[str, ViewRecord] = {}  # image key -> view record
        self.anchor_views: Dict[str, ViewRecord] = {}  # image key -> anchor view
        self.observations: Dict[str, np.ndarray] = {}  # image key -> RGB image

        self.clustering = SceneHierarchicalClustering(min_sample_split=0, random_state=66)

        self.detection_model = detection_model
        self.detection_model.set_classes(self.obj_classes.get_classes_arr())
        self.sam_predictor = sam_predictor
        self.clip_model = clip_model.to(self.device)
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer

    def close(self):
        if self.simulator is not None:
            self.simulator.close()
            self.simulator = None

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def get_observation(self, pts, angle=None, rotation=None):
        assert (angle is None) ^ (rotation is None), "Only one of angle and rotation should be specified"
        agent_state = habitat_sim.AgentState()
        agent_state.position = pts
        agent_state.rotation = get_quaternion(angle, 0) if angle is not None else rotation
        self.agent.set_state(agent_state)
        obs = self.simulator.get_sensor_observations()

        sensor = self.agent.get_state().sensor_states["depth_sensor"]
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(sensor.rotation)
        cam_pose[:3, 3] = sensor.position
        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])
        return obs, cam_pose

    def get_frontier_observation(self, pts, view_dir, camera_tilt=0.0):
        """Image captured at ``pts`` facing a frontier region."""
        agent_state = habitat_sim.AgentState()
        default_view_dir = np.asarray([0.0, 0.0, -1.0])
        if np.linalg.norm(view_dir) < 1e-3:
            view_dir = default_view_dir
        view_dir = view_dir / np.linalg.norm(view_dir)
        agent_state.position = pts
        if np.dot(view_dir, default_view_dir) < -1 + 1e-3:
            # a 180-degree turn is not unique; rotate around the y-axis
            agent_state.rotation = quat_to_coeffs(
                quaternion.quaternion(0, 0, 1, 0) * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()
        else:
            agent_state.rotation = quat_to_coeffs(
                quat_from_two_vectors(default_view_dir, view_dir)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()
        self.agent.set_state(agent_state)
        obs = self.simulator.get_sensor_observations()
        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])
        return obs

    # ------------------------------------------------------------------ #
    # Object instances
    # ------------------------------------------------------------------ #
    def update_objects(
        self,
        image_rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics,
        cam_pos,
        pts,
        pts_voxel,
        img_path: str,
    ) -> Tuple[np.ndarray, List[int]]:
        """Detect, segment and merge the object instances of one observation.

        Returns the annotated image and the ids of the newly created instances.
        """
        obj_classes = self.obj_classes

        results = self.detection_model.predict(image_rgb, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detection_class_labels = [
            f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(detection_class_ids)
        ]
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        if xyxy_tensor.numel() != 0:
            sam_out = self.sam_predictor.predict(image_rgb, bboxes=xyxy_tensor, verbose=False)
            masks_np = sam_out[0].masks.data.cpu().numpy()
        else:
            masks_np = np.empty((0, *image_rgb.shape[:2]), dtype=np.float64)

        curr_det = sv.Detections(xyxy=xyxy_np, confidence=confidences, class_id=detection_class_ids, mask=masks_np)
        if len(curr_det) == 0:
            return image_rgb, []

        # remove overlapping detections
        curr_det, _ = filter_detections(
            image=image_rgb,
            detections=curr_det,
            classes=obj_classes,
            given_labels=detection_class_labels,
            iou_threshold=self.cfg_cg.object_detection_iou_threshold,
            min_mask_size_ratio=self.cfg_cg.min_mask_size_ratio,
            confidence_threshold=self.cfg_cg.object_detection_confidence_threshold,
        )
        if curr_det is None:
            return image_rgb, []

        image_crops, image_feats, text_feats = compute_clip_features_batched(
            image_rgb,
            curr_det,
            self.clip_model,
            self.clip_preprocess,
            self.clip_tokenizer,
            obj_classes.get_classes_arr(),
            self.device,
        )
        raw_gobs = {
            "xyxy": curr_det.xyxy,
            "confidence": curr_det.confidence,
            "class_id": curr_det.class_id,
            "mask": curr_det.mask,
            "classes": obj_classes.get_classes_arr(),
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "detection_class_labels": detection_class_labels,
        }
        gobs = filter_gobs(
            resize_gobs(raw_gobs, image_rgb),
            image_rgb,
            skip_bg=self.cfg_cg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=self.cfg_cg.mask_area_threshold,
            max_bbox_area_ratio=self.cfg_cg.max_bbox_area_ratio,
            mask_conf_threshold=self.cfg_cg.mask_conf_threshold,
        )
        if len(gobs["mask"]) == 0:
            return image_rgb, []

        # keep e.g. pillows on couches as separate objects
        gobs["mask"] = mask_subtract_contained(gobs["xyxy"], gobs["mask"])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth,
            masks=gobs["mask"],
            cam_K=intrinsics[:3, :3],
            image_rgb=image_rgb,
            trans_pose=cam_pos,
            min_points_threshold=self.cfg_cg.min_points_threshold,
            spatial_sim_type=self.cfg_cg.spatial_sim_type,
            obj_pcd_max_points=self.cfg_cg.obj_pcd_max_points,
            device=self.device,
        )
        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg_cg["dbscan_eps"],
                    dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(spatial_sim_type=self.cfg_cg["spatial_sim_type"], pcd=obj["pcd"])
        if all(obj is None for obj in obj_pcds_and_bboxes):
            return image_rgb, []

        gobs["bbox"] = [obj["bbox"] if obj is not None else None for obj in obj_pcds_and_bboxes]
        gobs["pcd"] = [obj["pcd"] if obj is not None else None for obj in obj_pcds_and_bboxes]

        # visible objects are limited to the visibility distance of the view
        gobs = self._filter_by_distance(pts, gobs)
        detection_list = self._make_detection_list(gobs, obj_classes)
        if len(detection_list) == 0:
            return image_rgb, []

        view = ViewRecord(
            image=img_path,
            color=(random.random(), random.random(), random.random()),
            obs_point=pts_voxel,
            full_obj_list={obj_id: detection_list[obj_id]["conf"] for obj_id in detection_list.keys()},
        )
        det_visual_prompt = sv.Detections(xyxy=gobs["xyxy"], class_id=gobs["class_id"])

        if len(self.objects) == 0:
            self.objects.update(detection_list)
            det_visual_prompt.data["obj_id"] = list(detection_list.keys())
            view.visual_prompt = det_visual_prompt
            self.views[img_path] = view
            return image_rgb, list(detection_list.keys())

        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=self.cfg_cg["spatial_sim_type"],
            detection_list=detection_list,
            objects=self.objects,
            downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
        )
        visual_sim = compute_visual_similarities(detection_list, self.objects)
        agg_sim = aggregate_similarities(
            match_method=self.cfg_cg["match_method"],
            phys_bias=self.cfg_cg["phys_bias"],
            spatial_sim=spatial_sim,
            visual_sim=visual_sim,
        )
        match_indices = match_detections_to_objects(
            agg_sim=agg_sim,
            detection_threshold=self.cfg_cg["sim_threshold"],
            existing_obj_ids=list(self.objects.keys()),
            detected_obj_ids=list(detection_list.keys()),
        )
        captions, added_obj_ids, all_obj_ids = self._merge_matches(detection_list, match_indices, obj_classes, view)
        det_visual_prompt.data["obj_id"] = all_obj_ids
        view.visual_prompt = det_visual_prompt
        self.views[img_path] = view

        det_visualize = sv.Detections(xyxy=gobs["xyxy"], confidence=gobs["confidence"], class_id=gobs["class_id"])
        det_visualize.data["class_name"] = captions
        annotated = sv.BoundingBoxAnnotator(thickness=1).annotate(image_rgb.copy(), det_visualize)
        annotated = sv.LabelAnnotator(text_thickness=1, text_scale=0.25, text_color=sv.Color.BLACK).annotate(
            annotated, det_visualize
        )
        return annotated, added_obj_ids

    def _filter_by_distance(self, pts, gobs):
        keep = [
            idx
            for idx in range(len(gobs["bbox"]))
            if gobs["bbox"][idx] is not None
            and np.linalg.norm(gobs["bbox"][idx].center[[0, 2]] - pts[[0, 2]]) <= self.cfg.perception.object_visibility_dist_m
        ]
        for attribute in gobs.keys():
            if isinstance(gobs[attribute], str) or attribute in ("classes", "labels", "edges", "text_feats", "captions"):
                continue
            if isinstance(gobs[attribute], list):
                gobs[attribute] = [gobs[attribute][i] for i in keep]
            elif isinstance(gobs[attribute], np.ndarray):
                gobs[attribute] = gobs[attribute][keep]
            else:
                raise NotImplementedError(f"Unhandled type {type(gobs[attribute])}")
        return gobs

    def _merge_matches(self, detection_list: DetectionDict, match_indices, obj_classes, view: ViewRecord):
        captions, all_obj_ids, added_obj_ids = [], [], []
        for detected_obj_id, existing_obj_id in match_indices:
            if existing_obj_id is None:
                self.objects[detected_obj_id] = detection_list[detected_obj_id]
                obj = self.objects[detected_obj_id]
                captions.append(f"{detected_obj_id} {obj['class_name']} {obj['conf']:.3f} N")
                all_obj_ids.append(detected_obj_id)
                added_obj_ids.append(detected_obj_id)
                continue
            detected_obj = detection_list[detected_obj_id]
            merged_obj = merge_obj2_into_obj1(
                obj1=self.objects[existing_obj_id],
                obj2=detected_obj,
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=self.cfg_cg["dbscan_eps"],
                dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                device=self.device,
                run_dbscan=False,
            )
            # the instance takes the most frequent category of its detections
            most_common_class_id = Counter(merged_obj["class_id"]).most_common(1)[0][0]
            merged_obj["class_name"] = obj_classes.get_classes_arr()[most_common_class_id]
            view.full_obj_list[existing_obj_id] = detected_obj["conf"]
            view.full_obj_list.pop(detected_obj_id)
            self.objects[existing_obj_id] = merged_obj
            captions.append(
                f"{existing_obj_id} {merged_obj['class_name']} {detected_obj['conf']:.3f} {merged_obj['num_detections']}"
            )
            all_obj_ids.append(existing_obj_id)
        return captions, added_obj_ids, all_obj_ids

    def _make_detection_list(self, gobs, obj_classes) -> DetectionDict:
        detection_list = DetectionDict()
        for mask_idx in range(len(gobs["mask"])):
            if gobs["pcd"][mask_idx] is None:
                continue
            class_name = gobs["classes"][gobs["class_id"][mask_idx]]
            detection_list[self.object_id_counter] = {
                "id": self.object_id_counter,
                "class_name": class_name,
                "class_id": [obj_classes.get_classes_arr().index(class_name)],
                "num_detections": 1,
                "conf": gobs["confidence"][mask_idx],
                "pcd": gobs["pcd"][mask_idx],
                "bbox": gobs["bbox"][mask_idx],
                "clip_ft": to_tensor(gobs["image_feats"][mask_idx]),
                "image": None,  # key of the anchor view that holds the instance
            }
            self.object_id_counter += 1
        return detection_list

    def periodic_cleanup_objects(self, frame_idx, pts):
        """Denoise, filter and merge instances at the intervals of the ConceptGraphs config."""
        if processing_needed(self.cfg_cg["denoise_interval"], self.cfg_cg["run_denoise_final_frame"], frame_idx, is_final_frame=False):
            self.objects = measure_time(denoise_objects)(
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=self.cfg_cg["dbscan_eps"],
                dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                device=self.device,
                objects=self.objects,
            )
        if processing_needed(self.cfg_cg["filter_interval"], self.cfg_cg["run_filter_final_frame"], frame_idx, is_final_frame=False):
            self.objects = filter_objects(
                obj_min_points=self.cfg_cg["obj_min_points"],
                obj_min_detections=self.cfg_cg["obj_min_detections"],
                min_distance=self.cfg.perception.object_visibility_dist_m,
                objects=self.objects,
                pts=pts,
            )
        if processing_needed(self.cfg_cg["merge_interval"], self.cfg_cg["run_merge_final_frame"], frame_idx, is_final_frame=False):
            self.objects = measure_time(merge_objects)(
                merge_overlap_thresh=self.cfg_cg["merge_overlap_thresh"],
                merge_visual_sim_thresh=self.cfg_cg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=self.cfg_cg["merge_text_sim_thresh"],
                objects=self.objects,
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=self.cfg_cg["dbscan_eps"],
                dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                device=self.device,
            )

        # drop removed instances from the view records
        for key in list(self.views.keys()):
            view = self.views[key]
            view.cluster = [obj_id for obj_id in view.cluster if obj_id in self.objects]
            view.full_obj_list = {k: v for k, v in view.full_obj_list.items() if k in self.objects}
            if len(view.full_obj_list) == 0:
                self.views.pop(key)

    # ------------------------------------------------------------------ #
    # Anchor views
    # ------------------------------------------------------------------ #
    def _drop_empty_views(self):
        self.views = {k: v for k, v in self.views.items() if len(v.full_obj_list) > 0}
        self.anchor_views = {k: v for k, v in self.anchor_views.items() if len(v.cluster) > 0}

    def update_anchor_views(self, obj_ids, min_detection=1):
        """Re-cluster the instances near the agent and select the anchor views
        that observe each cluster (co-visibility clustering)."""
        self._drop_empty_views()
        prev_anchor_views = copy.deepcopy(self.anchor_views)

        obj_ids_temp = obj_ids.copy()
        for key, view in self.anchor_views.items():
            if any(obj_id in obj_ids_temp for obj_id in view.cluster):
                obj_ids = obj_ids.union(set(view.cluster))
                prev_anchor_views.pop(key)
        obj_ids = list(set(obj_ids))

        # instances observed fewer than ``min_detection`` times are not assigned
        obj_exclude = [obj_id for obj_id in self.objects.keys() if self.objects[obj_id]["num_detections"] < min_detection]
        obj_ids = [obj_id for obj_id in obj_ids if obj_id not in obj_exclude]

        obj_centers = np.zeros((len(obj_ids), 2))
        for i, obj_id in enumerate(obj_ids):
            obj_centers[i] = self.objects[obj_id]["bbox"].center[[0, 2]]
        if len(obj_centers) == 0:
            return

        new_anchor_views = self.clustering.fit(obj_centers, obj_ids, self.views)
        assert {o for v in new_anchor_views.values() for o in v.cluster} == set(obj_ids)

        for key, view in new_anchor_views.items():
            if key in prev_anchor_views:
                prev_anchor_views[key].cluster += view.cluster
            else:
                prev_anchor_views[key] = view
        self.anchor_views = prev_anchor_views

        for key, view in self.anchor_views.items():
            for obj_id in view.cluster:
                self.objects[obj_id]["image"] = key
        # the anchor views and their view records are the same objects
        for key, view in self.anchor_views.items():
            self.views[key] = view

    def visible_objects(self, view_key: str) -> List[int]:
        """O^vis(v): instances detected in the view within the visibility distance."""
        view = self.views.get(view_key) or self.anchor_views.get(view_key)
        if view is None:
            return []
        return [obj_id for obj_id in view.full_obj_list if obj_id in self.objects]

    def object_crop(self, view_key: str, obj_id: int, size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Crop of an instance from the image of a view, using the stored detection box."""
        view = self.views.get(view_key) or self.anchor_views.get(view_key)
        img = self.observations.get(view_key)
        if view is None or img is None or view.visual_prompt is None:
            return None
        ids = list(view.visual_prompt.data.get("obj_id", []))
        if obj_id not in ids:
            return None
        x0, y0, x1, y1 = view.visual_prompt.xyxy[ids.index(obj_id)].astype(int)
        crop = img[max(0, y0) : max(y0 + 1, y1), max(0, x0) : max(x0 + 1, x1)]
        if size is not None and crop.size > 0:
            from evomemnav.env.habitat_utils import resize_image

            h, w = crop.shape[:2]
            scale = min(size[0] / h, size[1] / w)
            crop = resize_image(crop, max(1, int(h * scale)), max(1, int(w * scale)))
        return crop if crop.size > 0 else None
