"""Configuration loading.

A config file may name a parent with ``_base_: <relative path>``; its keys are
merged on top of the parent. Command line overrides use the OmegaConf dot-list
syntax, e.g. ``coarse.anchor_budget=5``.
"""

from pathlib import Path
from typing import Iterable, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(path, overrides: Optional[Iterable[str]] = None) -> DictConfig:
    cfg = _load_with_base(Path(path).resolve())
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def _load_with_base(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    base = cfg.pop("_base_", None)
    if base is None:
        return cfg
    return OmegaConf.merge(_load_with_base((path.parent / str(base)).resolve()), cfg)
