"""Visual-Semantic Memory Graph (VSMGraph), Section 3.2.

G = (R, V, O, E) with a room-view-object hierarchy:
  * view nodes keep observed images with their poses: a view node is added at
    each visited pose and linked to the previous one by a navigability edge;
    anchor views record object-rich observations and frontier views face the
    unexplored regions;
  * object nodes are the merged object instances (``Scene.objects``) and are
    linked to the anchor views that observe them by visibility edges;
  * room nodes group the views and objects with the same room label, where the
    room label of a view is the room type whose CLIP text embedding best matches
    the image of the view.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ViewNode:
    id: str
    kind: str  # "visited", "anchor" or "frontier"
    position: np.ndarray  # habitat-frame position of the view pose
    image_key: Optional[str] = None
    room: str = "unknown"
    objects: List[int] = field(default_factory=list)  # visible objects (anchor views)
    anchor_link: Optional[str] = None  # nearest visited view (frontier views)


class RoomLabeler:
    """Threshold-free CLIP room label: argmax over 'a photo of a <room>' prompts."""

    def __init__(self, clip_model, clip_preprocess, clip_tokenizer, room_vocab: Sequence[str], image_size: int = 360):
        import torch

        self.torch = torch
        self.model = clip_model
        self.preprocess = clip_preprocess
        self.vocab = list(room_vocab)
        self.image_size = int(image_size)
        self.device = next(clip_model.parameters()).device
        with torch.no_grad():
            tokens = clip_tokenizer([f"a photo of a {r}" for r in self.vocab]).to(self.device)
            feat = clip_model.encode_text(tokens)
            self.text_feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)

    def __call__(self, image: Optional[np.ndarray]) -> str:
        if image is None:
            return "unknown"
        from PIL import Image

        from evomemnav.env.habitat_utils import resize_image

        img = resize_image(np.asarray(image)[..., :3], self.image_size, self.image_size)
        with self.torch.no_grad():
            x = self.preprocess(Image.fromarray(img)).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(x)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            idx = int((feat @ self.text_feat.T).squeeze(0).argmax().item())
        return self.vocab[idx]


class VSMGraph:
    def __init__(self, room_labeler: Optional[Callable[[Optional[np.ndarray]], str]] = None):
        self.room_labeler = room_labeler
        self._room_cache: Dict[Tuple, str] = {}
        self.reset()

    def reset(self):
        """The VSMGraph is cleared at the end of each episode."""
        self.visited: List[ViewNode] = []
        self.navigability: List[Tuple[str, str]] = []
        self.anchors: Dict[str, ViewNode] = {}
        self.frontiers: Dict[str, ViewNode] = {}
        self.objects: Dict[int, Dict] = {}
        self._room_cache.clear()

    # ------------------------------------------------------------------ build
    def _label(self, cache_key: Tuple, image: Optional[np.ndarray]) -> str:
        if self.room_labeler is None:
            return "unknown"
        if cache_key not in self._room_cache:
            self._room_cache[cache_key] = self.room_labeler(image)
        return self._room_cache[cache_key]

    def add_visited_view(self, step: int, position: np.ndarray, image_key: Optional[str]) -> ViewNode:
        node = ViewNode(id=f"v{step}", kind="visited", position=np.asarray(position, dtype=float), image_key=image_key)
        if self.visited:
            self.navigability.append((self.visited[-1].id, node.id))
        self.visited.append(node)
        return node

    def sync_anchor_views(self, scene, voxel2habitat: Callable) -> None:
        nodes = {}
        for key, view in scene.anchor_views.items():
            nodes[key] = ViewNode(
                id=key,
                kind="anchor",
                position=np.asarray(voxel2habitat(np.asarray(view.obs_point)[:2]), dtype=float),
                image_key=key,
                room=self._label(("anchor", key), scene.observations.get(key)),
                objects=scene.visible_objects(key),
            )
        self.anchors = nodes
        self.objects = {
            int(oid): {"category": obj["class_name"], "anchor": obj.get("image")} for oid, obj in scene.objects.items()
        }

    def sync_frontier_views(self, frontiers: Sequence, voxel2habitat: Callable) -> List[ViewNode]:
        """Returns the frontier view nodes in the order of ``frontiers``."""
        nodes, ordered = {}, []
        for ft in frontiers:
            vid = f"f{ft.frontier_id}"
            node = ViewNode(
                id=vid,
                kind="frontier",
                position=np.asarray(voxel2habitat(ft.position), dtype=float),
                image_key=ft.image,
                room=self._label(("frontier", ft.frontier_id, ft.image), ft.feature),
                anchor_link=self.nearest_visited(voxel2habitat(ft.position)),
            )
            nodes[vid] = node
            ordered.append(node)
        self.frontiers = nodes
        return ordered

    def nearest_visited(self, position) -> Optional[str]:
        if not self.visited:
            return None
        p = np.asarray(position, dtype=float)
        d = [np.linalg.norm((n.position - p)[[0, 2]]) for n in self.visited]
        return self.visited[int(np.argmin(d))].id

    # ---------------------------------------------------------------- queries
    def visibility_edges(self) -> List[Tuple[str, int]]:
        return [(a.id, oid) for a in self.anchors.values() for oid in a.objects]

    def rooms(self) -> Dict[str, Dict[str, list]]:
        """Room nodes: room label -> anchor views, frontier views and objects."""
        out: Dict[str, Dict[str, list]] = {}
        for a in self.anchors.values():
            out.setdefault(a.room, {"anchors": [], "frontiers": [], "objects": []})["anchors"].append(a.id)
        for f in self.frontiers.values():
            out.setdefault(f.room, {"anchors": [], "frontiers": [], "objects": []})["frontiers"].append(f.id)
        for oid, obj in self.objects.items():
            anchor = self.anchors.get(obj.get("anchor"))
            if anchor is not None:
                out[anchor.room]["objects"].append(oid)
        return out

    def summary(self, room_hints: Sequence[str] = (), category_hints: Sequence[str] = ()) -> str:
        """Compact topo-memory summary given to candidate selection."""
        lines = ["=== Topo Memory (compact) ==="]
        hinted = [r for r in room_hints if r and r != "unknown"]
        if hinted:
            lines.append(f"Room focus: {hinted}")
        if category_hints:
            lines.append(f"Category focus: {list(category_hints)[:8]}")
        rooms = self.rooms()
        present = sorted(rooms.items(), key=lambda kv: -(len(kv[1]["anchors"]) + len(kv[1]["frontiers"])))
        if present:
            lines.append(
                "Rooms(now): "
                + "; ".join(
                    f"{room}:A{len(v['anchors'])}/F{len(v['frontiers'])}/O{len(v['objects'])}" for room, v in present[:8]
                )
            )
        return "\n".join(lines)
