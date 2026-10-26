"""View records of the observation memory."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ViewRecord:
    """An observed image with its detections.

    Every observation that contains detections is kept as a view record. Anchor
    views are the records selected by co-visibility clustering of the object
    instances (``memory/view_clustering.py``); for those, ``cluster`` lists the
    instances assigned to the view.
    """

    image: str  # key of the image in Scene.observations
    color: Tuple[float, float, float]  # display color for visualization
    obs_point: np.ndarray  # voxel position of the pose where the image was taken
    full_obj_list: Dict[int, float] = field(default_factory=dict)  # visible objects -> confidence
    cluster: List[int] = field(default_factory=list)
    position: Optional[np.ndarray] = None  # voxel center of the navigation target, set by the planner
    visual_prompt: Any = None  # supervision.Detections with an "obj_id" field

    def __eq__(self, other):
        raise NotImplementedError("View records are compared by their image key.")
