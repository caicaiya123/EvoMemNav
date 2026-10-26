# Adapted from ConceptGraphs (https://github.com/concept-graphs/concept-graphs), MIT License.
# Only the functions used by EvoMemNav are kept.

import logging
from pathlib import Path

import numpy as np
import supervision as sv
import time


def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


def filter_detections(
    image,
    detections: sv.Detections,
    classes,
    top_x_detections=None,
    confidence_threshold: float = 0.0,
    given_labels=None,
    iou_threshold: float = 0.80,  # IoU similarity threshold
    proximity_threshold: float = 20.0,  # Default proximity threshold
    keep_larger: bool = True,  # Keep the larger bounding box by area if True, else keep the smaller
    min_mask_size_ratio=0.00025,
) -> tuple[sv.Detections, list[str]]:
    """
    Filter detections based on confidence, top X detections, and proximity of bounding boxes.
    Args:
        proximity_threshold (float): The minimum distance between centers of bounding boxes to consider them non-overlapping.
        keep_larger (bool): If True, keeps the larger bounding box when overlaps occur; otherwise keeps the smaller.
    Returns:
        tuple[sv.Detections, list[str]]: Filtered detections and labels.
    """
    if not (
        hasattr(detections, "confidence")
        and hasattr(detections, "class_id")
        and hasattr(detections, "xyxy")
    ):
        print("Detections object is missing required attributes.")
        return detections, []

    # Sort by confidence initially
    detections_combined = sorted(
        zip(
            detections.confidence,
            detections.class_id,
            detections.xyxy,
            detections.mask,
            range(len(given_labels)),
        ),
        key=lambda x: x[0],
        reverse=True,
    )

    if top_x_detections is not None:
        detections_combined = detections_combined[:top_x_detections]

    # Calculate the total number of pixels as a threshold for small masks
    total_pixels = image.shape[0] * image.shape[1]
    small_mask_size = total_pixels * min_mask_size_ratio

    # Further filter based on proximity
    filtered_detections = []
    for idx, current_det in enumerate(detections_combined):
        conf, curr_class_id, curr_xyxy, curr_mask, _ = current_det
        curr_center = (
            (curr_xyxy[0] + curr_xyxy[2]) / 2,
            (curr_xyxy[1] + curr_xyxy[3]) / 2,
        )
        curr_area = (curr_xyxy[2] - curr_xyxy[0]) * (curr_xyxy[3] - curr_xyxy[1])
        keep = True

        # check mask size and remove if too small
        mask_size = np.count_nonzero(current_det[3])
        if mask_size < small_mask_size:
            logging.debug(
                f"Removing {classes.get_classes_arr()[curr_class_id]} because the mask size is too small."
            )
            keep = False
            continue

        # Check confidence threshold
        if conf < confidence_threshold:
            logging.debug(
                f"Removing {classes.get_classes_arr()[curr_class_id]} because it has a confidence of {conf} < {confidence_threshold}."
            )
            keep = False
            continue

        for other in filtered_detections:
            _, other_class_id, other_xyxy, other_mask, _ = other

            if mask_iou(curr_mask, other_mask) > iou_threshold:
                keep = False
                logging.debug(
                    f"Removing {classes.get_classes_arr()[curr_class_id]} because it has an IoU of {mask_iou(curr_mask, other_mask)} with object {classes.get_classes_arr()[other_class_id]}."
                )
                break

            other_center = (
                (other_xyxy[0] + other_xyxy[2]) / 2,
                (other_xyxy[1] + other_xyxy[3]) / 2,
            )
            other_area = (other_xyxy[2] - other_xyxy[0]) * (
                other_xyxy[3] - other_xyxy[1]
            )

            # Calculate distance between centers
            dist = np.sqrt(
                (curr_center[0] - other_center[0]) ** 2
                + (curr_center[1] - other_center[1]) ** 2
            )
            if dist < proximity_threshold:
                if (keep_larger and curr_area > other_area) or (
                    not keep_larger and curr_area < other_area
                ):
                    filtered_detections.remove(other)
                else:
                    keep = False
                    break
        # print(given_labels[idx])
        if classes.get_classes_arr()[curr_class_id] in classes.bg_classes:
            logging.debug(
                f"Removing {classes.get_classes_arr()[curr_class_id]} because it is a background class, specifically {classes.bg_classes}."
            )
            keep = False
            continue

        if keep:
            filtered_detections.append(current_det)

    if len(filtered_detections) == 0:
        return None, None

    # Unzip the filtered results
    confidences, class_ids, xyxy, masks, indices = zip(*filtered_detections)
    filtered_labels = [given_labels[i] for i in indices]

    # Create new detections object
    filtered_detections = sv.Detections(
        class_id=np.array(class_ids, dtype=np.int64),
        confidence=np.array(confidences, dtype=np.float32),
        xyxy=np.array(xyxy, dtype=np.float32),
        mask=np.array(masks, dtype=np.bool_),
    )

    return filtered_detections, filtered_labels


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        # print(f"Starting {func.__name__}...")
        result = func(
            *args, **kwargs
        )  # Call the function with any arguments it was called with
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(
            f"Done! Execution time of {func.__name__} function: {elapsed_time:.2f} seconds"
        )
        return result  # Return the result of the function call

    return wrapper


class ObjectClasses:
    """Fixed open vocabulary of the detector, read from a text file with one class per line.

    EvoMemNav uses the ScanNet-200 class list; class order follows the file.
    """

    def __init__(self, classes_file_path, bg_classes):
        self.classes_file_path = Path(classes_file_path)
        self.bg_classes = bg_classes
        with open(self.classes_file_path, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        seen = set()
        self.classes = [c for c in names if c != "unknown" and not (c in seen or seen.add(c))]
        logging.info(f"Loaded {len(self.classes)} classes from {self.classes_file_path}")

    def get_classes_arr(self):
        return self.classes

    def get_bg_classes_arr(self):
        return self.bg_classes
