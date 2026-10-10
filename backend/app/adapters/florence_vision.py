"""Describe images with Microsoft Florence-2 (runs inside our Python process on the GPU).

Why Florence-2 instead of Qwen2.5-VL via Ollama: on a 4 GB GPU Qwen 3B spilled
to the CPU (~29 s/frame). Florence-2-large is ~0.8B parameters (~1.5 GB in fp16),
fits easily, and does three things we need, quickly:
    <MORE_DETAILED_CAPTION>  a paragraph describing the frame
    <OCR>                    text visible on screen (subtitles, signs)
    <OD>                     object labels (can be disabled: MOVIEQA_FLORENCE_OBJECTS=0)
It has no "mood" or structured "people" output; the scene stage's LLM infers those later.

Speed on an RTX 3050 4 GB (measured, one frame): caption 8 s (beams=1) / 9.6 s (beams=3),
OCR 3.4 s, objects 5.2 s. Each GPU call has a large fixed cost, so frames are processed
in batches (config.caption_batch_size). Batched, all three tasks together: 1.8 s/frame.

A CLASS because it holds the loaded model. Same interface as OllamaVision:
    describe(image_path) -> dict,  unload()
Uses the native Florence-2 support in transformers >= 5 (no trust_remote_code).
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TASK_CAPTION = "<MORE_DETAILED_CAPTION>"
TASK_OCR = "<OCR>"
TASK_OBJECTS = "<OD>"


class FlorenceVision:
    def __init__(self, model_id: str = "florence-community/Florence-2-large", device: str = "auto",
                 with_objects: bool = False, caption_beams: int = 1) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                f"Florence-2 could not be imported ({exc}). It needs torch and transformers>=5 in the "
                "project venv: pip install torch --index-url https://download.pytorch.org/whl/cu128 "
                "then pip install -r requirements.txt"
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.with_objects = with_objects
        self.caption_beams = caption_beams
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.device == "cpu":
            log.warning("Florence-2 is running on the CPU (no CUDA torch found) — this will be slow. "
                        "Install CUDA torch: pip install torch --index-url https://download.pytorch.org/whl/cu128")

        log.info("loading %s on %s (%s) — first run downloads the model", model_id, self.device, self.dtype)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Florence2ForConditionalGeneration.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(self.device).eval()

    # ---------------------------------------------------------------- main call
    def describe(self, image: Path) -> dict[str, Any]:
        return self.describe_batch([image])[0]

    def describe_batch(self, images: list[Path]) -> list[dict[str, Any]]:
        """Describe several frames with one GPU call per task (much faster than one by one,
        because each call has a large fixed cost on a small GPU)."""
        from PIL import Image

        imgs = []
        for path in images:
            with Image.open(path) as im:
                imgs.append(im.convert("RGB"))

        captions = self._run(imgs, TASK_CAPTION, max_new_tokens=256, num_beams=self.caption_beams)
        ocrs = self._run(imgs, TASK_OCR, max_new_tokens=128, num_beams=1)
        ods = self._run(imgs, TASK_OBJECTS, max_new_tokens=256, num_beams=1) if self.with_objects else [None] * len(imgs)

        results = []
        for caption, ocr, od in zip(captions, ocrs, ods):
            labels = od.get("labels", []) if isinstance(od, dict) else []
            results.append({
                "caption": caption.strip() if isinstance(caption, str) else "",
                "on_screen_text": _clean_ocr(ocr),
                "objects": list(dict.fromkeys(l.strip().lower() for l in labels if l and l.strip()))[:10],
            })
        return results

    def unload(self) -> None:
        """Free GPU memory for the next stage."""
        self.model = None
        self.processor = None
        import gc
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # ---------------------------------------------------------------- internals
    def _run(self, imgs: list, task: str, *, max_new_tokens: int, num_beams: int) -> list:
        torch = self._torch
        inputs = self.processor(text=[task] * len(imgs), images=imgs, return_tensors="pt",
                                padding=True).to(self.device, self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      num_beams=num_beams, do_sample=False)
        texts = self.processor.batch_decode(ids, skip_special_tokens=False)
        out = []
        for text, img in zip(texts, imgs):
            text = text.replace("<pad>", "")          # shorter answers in a batch are padded
            parsed = self.processor.post_process_generation(text, task=task, image_size=(img.width, img.height))
            out.append(parsed.get(task, "") if isinstance(parsed, dict) else parsed)
        return out


_WORD = re.compile(r"[A-Za-z]{2,}")


def _clean_ocr(text: Any) -> str:
    """Florence's OCR returns junk on frames without text (e.g. 'S' or '.'). Keep it only if it
    looks like real text: at least two words of 2+ letters, or one word of 4+ letters."""
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text.replace("<pad>", "").replace("</s>", "").replace("<s>", "")).strip()
    words = _WORD.findall(t)
    if len(words) >= 2 or any(len(w) >= 4 for w in words):
        return t
    return ""
