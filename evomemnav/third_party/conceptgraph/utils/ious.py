# Adapted from ConceptGraphs (https://github.com/concept-graphs/concept-graphs), MIT License.
# Only the functions used by EvoMemNav are kept.

import numpy as np
import torch

import open3d as o3d


def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of axis-aligned 3D bounding boxes.

    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)

    returns: (M, N)
    """
    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1)  # Shape: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1)  # Shape: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1)  # Shape: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1)  # Shape: (N, 3)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

    # Compute volume of intersection box
    inter_vol = torch.prod(
        torch.clamp(inter_max - inter_min, min=0), dim=2
    )  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

    return iou


def compute_giou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    Compute the generalized IoU between two sets of axis-aligned 3D bounding boxes.

    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)

    returns: (M, N)
    """
    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1)  # Shape: (M, D)
    bbox1_max, _ = bbox1.max(dim=1)  # Shape: (M, D)
    bbox2_min, _ = bbox2.min(dim=1)  # Shape: (N, D)
    bbox2_max, _ = bbox2.max(dim=1)  # Shape: (N, D)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, D)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, D)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, D)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, D)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, D)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, D)

    # to obtain the coordinates of enclosing box
    enclosing_min = torch.min(bbox1_min, bbox2_min)  # Shape: (M, N, D)
    enclosing_max = torch.max(bbox1_max, bbox2_max)  # Shape: (M, N, D)

    # Compute volume of intersection box
    inter_vol = torch.prod(
        torch.clamp(inter_max - inter_min, min=0), dim=2
    )  # Shape: (M, N)
    enclosing_vol = torch.prod(enclosing_max - enclosing_min, dim=2)  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)
    union_vol = bbox1_vol + bbox2_vol - inter_vol

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (union_vol + 1e-10)
    giou = iou - (enclosing_vol - union_vol) / (enclosing_vol + 1e-10)

    return giou


def compute_3d_iou_accurate_batch(bbox1, bbox2):
    """
    Compute IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.

    bbox1: (M, 8, D), e.g. (M, 8, 3)
    bbox2: (N, 8, D), e.g. (N, 8, 3)

    returns: (M, N)
    """
    # Must expend the box beforehand, otherwise it may results overestimated results
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)

    import pytorch3d.ops as ops

    bbox1 = bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    bbox2 = bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]]

    inter_vol, iou = ops.box3d_overlap(bbox1.float(), bbox2.float())

    return iou


def compute_3d_box_volume_batch(bbox: torch.Tensor) -> torch.Tensor:
    """
    Compute the volume of a set of rectangular boxes.
    This assumes bbox corner order follows the open3d convention, which is:
    ---, +--, -+-, --+, +++, -++, +-+, ++-
    See https://github.com/isl-org/Open3D/blob/47f4ee936841ae9f8a9c4ce5d9162bd5b3e0279f/cpp/open3d/geometry/BoundingVolume.cpp#L92

    bbox: (N, 8, D)

    returns: (N,)
    """
    a = torch.linalg.vector_norm(bbox[:, 0, :] - bbox[:, 1, :], ord=2, dim=1)
    b = torch.linalg.vector_norm(bbox[:, 0, :] - bbox[:, 2, :], ord=2, dim=1)
    c = torch.linalg.vector_norm(bbox[:, 0, :] - bbox[:, 3, :], ord=2, dim=1)

    vol = a * b * c
    return vol


def expand_3d_box(bbox: torch.Tensor, eps=0.02) -> torch.Tensor:
    """
    Expand the side of 3D boxes such that each side has at least eps length.
    Assumes the bbox cornder order in open3d convention.

    bbox: (N, 8, D)

    returns: (N, 8, D)
    """
    center = bbox.mean(dim=1)  # shape: (N, D)

    va = bbox[:, 1, :] - bbox[:, 0, :]  # shape: (N, D)
    vb = bbox[:, 2, :] - bbox[:, 0, :]  # shape: (N, D)
    vc = bbox[:, 3, :] - bbox[:, 0, :]  # shape: (N, D)

    a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)  # shape: (N, 1)

    va = torch.where(a < eps, va / a * eps, va)  # shape: (N, D)
    vb = torch.where(b < eps, vb / b * eps, vb)  # shape: (N, D)
    vc = torch.where(c < eps, vc / c * eps, vc)  # shape: (N, D)

    new_bbox = torch.stack(
        [
            center - va / 2.0 - vb / 2.0 - vc / 2.0,
            center + va / 2.0 - vb / 2.0 - vc / 2.0,
            center - va / 2.0 + vb / 2.0 - vc / 2.0,
            center - va / 2.0 - vb / 2.0 + vc / 2.0,
            center + va / 2.0 + vb / 2.0 + vc / 2.0,
            center - va / 2.0 + vb / 2.0 + vc / 2.0,
            center + va / 2.0 - vb / 2.0 + vc / 2.0,
            center + va / 2.0 + vb / 2.0 - vc / 2.0,
        ],
        dim=1,
    )  # shape: (N, 8, D)

    new_bbox = new_bbox.to(bbox.device)
    new_bbox = new_bbox.type(bbox.dtype)

    return new_bbox


def compute_enclosing_vol(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    Compute the enclosing volume between every pair of boxes in bbox1 and bbox2.
    This is an accurate but slow version using convex hull

    bbox1: (M, 8, D)
    bbox2: (N, 8, D)

    returns: (M, N)
    """
    M = bbox1.shape[0]
    N = bbox2.shape[0]

    enclosing_vol = torch.zeros((M, N), dtype=bbox1.dtype, device=bbox1.device)
    for i in range(bbox1.shape[0]):
        for j in range(bbox2.shape[0]):
            pcd_union = o3d.geometry.PointCloud()
            bbox_points_union = torch.cat([bbox1[i], bbox2[j]], dim=0)  # (16, 3)
            pcd_union.points = o3d.utility.Vector3dVector(
                bbox_points_union.cpu().numpy()
            )
            enclosing_mesh, _ = pcd_union.compute_convex_hull(joggle_inputs=True)
            try:
                enclosing_vol[i, j] = enclosing_mesh.get_volume()
            except:
                # This occurs commonly when the enclosing_mesh is not watertight.
                enclosing_mesh = pcd_union.get_axis_aligned_bounding_box()
                enclosing_vol[i, j] = enclosing_mesh.volume()

    return enclosing_vol


def compute_3d_giou_accurate_batch(
    bbox1: torch.Tensor, bbox2: torch.Tensor
) -> torch.Tensor:
    """
    Compute Generalized IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.

    bbox1: (M, 8, D), e.g. (M, 8, 3)
    bbox2: (N, 8, D), e.g. (N, 8, 3)

    returns: (M, N)
    """
    # Must expend the box beforehand, otherwise it may results overestimated results
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)

    bbox1_vol = compute_3d_box_volume_batch(bbox1)
    bbox2_vol = compute_3d_box_volume_batch(bbox2)

    import pytorch3d.ops as ops

    inter_vol, iou = ops.box3d_overlap(
        bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]].float(),
        bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]].float(),
    )
    union_vol = bbox1_vol.unsqueeze(1) + bbox2_vol.unsqueeze(0) - inter_vol

    enclosing_vol = compute_enclosing_vol(bbox1, bbox2)
    # enclosing_vol = compute_enclosing_vol_fast(bbox1, bbox2)

    giou = iou - (enclosing_vol - union_vol) / enclosing_vol

    return giou


def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    """
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.

    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2

    Returns:
        mask_sub: (N, H, W), binary mask
    """

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])  # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(
        xyxy[:, None, 2:], xyxy[None, :, 2:]
    )  # right-bottom points (N, N, 2)

    inter = (rb - lt).clip(
        min=0
    )  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

    inter_over_box1 = inter_areas / areas[:, None]  # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T  # (N, N)

    # if the intersection area is smaller than th2 of the area of box1,
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)  # (N, N)
    contained_idx = contained.nonzero()  # (num_contained, 2)

    mask_sub = mask.copy()  # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (
            ~mask_sub[contained_idx[1][i]]
        )

    return mask_sub
