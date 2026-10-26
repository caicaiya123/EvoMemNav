"""Goal keys and node keys of the experience memory.

Goal key levels (Appendix C of the supplementary file):
  L1 = (modality, target category, goal content)
  L2 = (modality, target category)
  L3 = (modality)
The node key is the cell of a fixed grid in the scene frame that contains the
position of the view.
"""

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence


def _norm(text) -> str:
    return " ".join(str(text or "").strip().lower().split())


@dataclass(frozen=True)
class GoalKey:
    modality: str
    category: str
    content: str = ""  # short digest of the goal content (description text or goal image)

    def levels(self) -> List[str]:
        m, c = _norm(self.modality), _norm(self.category)
        return [f"L1|{m}|{c}|{self.content}", f"L2|{m}|{c}", f"L3|{m}"]

    @property
    def l1(self) -> str:
        return self.levels()[0]


def goal_content_digest(modality: str, description: Optional[str] = None, image_bytes: Optional[bytes] = None) -> str:
    """Digest of the goal content: the description of a language goal, the
    pixels of an image goal, and nothing beyond the category for object goals."""
    m = _norm(modality)
    if m == "description" and description:
        return hashlib.sha1(_norm(description).encode("utf-8")).hexdigest()[:12]
    if m == "image" and image_bytes:
        return hashlib.sha1(image_bytes).hexdigest()[:12]
    return ""


def cell_key(position_xz: Sequence[float], cell_size_m: float) -> str:
    """Grid cell of a scene-frame position given as (x, z) in meters."""
    x, z = float(position_xz[0]), float(position_xz[1])
    return f"{int(x // cell_size_m)},{int(z // cell_size_m)}"
