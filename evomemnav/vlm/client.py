"""OpenAI-compatible client for the VLM (Qwen3-VL served by e.g. vLLM).

Every call is recorded with its kind (``room_hint``, ``selection`` or
``verification``), latency and token usage, so that the per-subtask VLM cost
can be reported.
"""

import base64
import logging
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

# A prompt part is either plain text or a (text, RGB image) pair.
Part = Union[str, Tuple[str, np.ndarray]]


def encode_image(img: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(np.asarray(img)[..., :3].astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def to_messages(system: str, parts: Sequence[Part]) -> List[Dict[str, Any]]:
    content = []
    for part in parts:
        text, image = (part, None) if isinstance(part, str) else part
        if text:
            content.append({"type": "text", "text": text})
        if image is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(image)}", "detail": "high"},
                }
            )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


class VLMClient:
    def __init__(self, cfg):
        from openai import OpenAI

        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=os.environ.get(cfg.api_key_env, "EMPTY"))
        self._records: List[Dict[str, Any]] = []

    def chat(self, system: str, parts: Sequence[Part], kind: str) -> Optional[str]:
        messages = to_messages(system, parts)
        n_images = sum(1 for p in parts if not isinstance(p, str) and p[1] is not None)
        for attempt in range(int(self.cfg.max_retries)):
            try:
                t0 = time.time()
                completion = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=float(self.cfg.temperature),
                    max_tokens=int(self.cfg.max_tokens),
                    seed=int(self.cfg.seed),
                )
                latency = time.time() - t0
            except Exception as e:  # network errors, rate limits, server restarts
                logging.warning(f"[VLM] {kind} call failed (attempt {attempt + 1}): {e}")
                time.sleep(3)
                continue
            usage = getattr(completion, "usage", None)
            self._records.append(
                {
                    "kind": kind,
                    "latency_s": latency,
                    "n_images": n_images,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            )
            return completion.choices[0].message.content
        return None

    def drain(self) -> List[Dict[str, Any]]:
        """Return and clear the call records collected since the last drain."""
        records, self._records = self._records, []
        return records
