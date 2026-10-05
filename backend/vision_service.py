"""
vision_service.py — Vision & OCR Engine for DeepFind's Tier-2 (idle-time) pipeline.

Consumes files handed off by the file watcher — plain images, or scanned PDFs
with no digital text layer — and produces OCR text + a CLIP image vector for
each one, in the same 512-dim space as the text-side Qdrant/clip-ViT-B-32-text
encoder.

Constraints this file is written against:
  - 8GB RAM, CPU-only, fully offline, ONNX Runtime only (no PyTorch)
  - process one item at a time; bounded working set (~150MB target)
  - both models loaded once at import/startup, reused for every call
  - scheduling (idle-check via should_pause_tier2()) lives in the caller,
    not here — this module is a pure "file in -> dict out" worker

CLIP models are fp32 as published (Qdrant/clip-ViT-B-32-vision /
-text, no quantization arg passed) to get the pipeline working end to end
first. If RAM pressure shows up during testing, the follow-up step is an
onnxruntime.quantization.quantize_dynamic() pass over the downloaded
model.onnx — not a fastembed model-name swap, since fastembed doesn't
expose a distinct INT8 variant for this model.

OPEN ITEMS TO VERIFY BEFORE SHIPPING (flagged inline too):
  1. Download the English-only RapidOCR model trio (det/rec/dict) — see
     OCR_MODELS_DIR below — the default pip install ships the larger
     Chinese+English ch_PP-OCRv4 bundle, which is NOT what gets loaded here.
  2. Confirm your installed fastembed version's ImageEmbedding constructor
     still accepts `threads=`; this has moved around across releases.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import pymupdf as fitz  # PyMuPDF's own package now prefers this import name over `import fitz`
import numpy as np
from fastembed import ImageEmbedding
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger("deepfind.vision_service")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS = {".pdf"}

RASTER_DPI = 170            # inside your 150-200 target band
TEXT_LAYER_MIN_CHARS = 20   # below this, a PDF page counts as "no text layer"
GC_EVERY_N_ITEMS = 15       # midpoint of your 10-20 target cadence

# English-only RapidOCR models. Download separately from the RapidOCR model
# releases (the "en_PP-OCRv3" det/rec pair + en_dict.txt) — do NOT rely on
# the multilingual bundle that ships by default with `pip install`.
OCR_MODELS_DIR = Path(os.environ.get("DEEPFIND_OCR_MODELS_DIR", "./models/ocr_en"))
OCR_ACCURACY_MODELS_DIR = Path(
    os.environ.get("DEEPFIND_OCR_ACCURACY_DIR", "./models/ocr_accuracy")
)
OCR_MODEL_PROFILES = {
    "speed": (
        str(OCR_MODELS_DIR / "en_PP-OCRv3_det_infer.onnx"),
        str(OCR_MODELS_DIR / "en_PP-OCRv3_rec_infer.onnx"),
        str(OCR_MODELS_DIR / "en_dict.txt"),
    ),
    "accuracy": (
        str(OCR_ACCURACY_MODELS_DIR / "ch_PP-OCRv4_det_infer.onnx"),
        str(OCR_ACCURACY_MODELS_DIR / "ch_PP-OCRv4_rec_infer.onnx"),
        str(OCR_ACCURACY_MODELS_DIR / "ch_dict.txt"),
    ),
}

CLIP_IMAGE_MODEL = "Qdrant/clip-ViT-B-32-vision"
# fastembed's own default cache lives inside the OS temp dir, which gets
# swept periodically — pin it to a project-local, persistent folder instead
# so there's one unambiguous place to pre-populate and copy to the offline
# machine.
CLIP_CACHE_DIR = Path(os.environ.get("DEEPFIND_CLIP_CACHE_DIR", "./models/clip_cache"))

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once, reused for every call
# ---------------------------------------------------------------------------

_ocr_engine: Optional[RapidOCR] = None
_clip_model: Optional[ImageEmbedding] = None
_items_processed = 0
_model_profile = "speed"


def set_model_profile(profile: str) -> None:
    """Select OCR ONNX models; keep CLIP paired with the text search encoder."""
    global _model_profile, _ocr_engine
    if profile not in OCR_MODEL_PROFILES:
        raise ValueError(f"Unsupported vision model profile: {profile}")
    if profile != _model_profile:
        _model_profile = profile
        _ocr_engine = None


def _get_ocr_engine() -> RapidOCR:
    global _ocr_engine
    if _ocr_engine is None:
        det_model_path, rec_model_path, rec_keys_path = OCR_MODEL_PROFILES[_model_profile]
        logger.info("Loading RapidOCR profile '%s' (capped threads)...", _model_profile)
        _ocr_engine = RapidOCR(
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            rec_keys_path=rec_keys_path,
            use_cls=False,  # skip angle classifier: fine for upright screenshots/
                            # scans, flip on if you expect rotated pages
            intra_op_num_threads=2,
            inter_op_num_threads=1,
            det_use_cuda=False,
            cls_use_cuda=False,
            rec_use_cuda=False,
        )
    return _ocr_engine


def _get_clip_model() -> ImageEmbedding:
    global _clip_model
    if _clip_model is None:
        logger.info("Loading CLIP image encoder (%s)...", CLIP_IMAGE_MODEL)
        _clip_model = ImageEmbedding(
            model_name=CLIP_IMAGE_MODEL,
            cache_dir=str(CLIP_CACHE_DIR),
            threads=2,  # verify this kwarg name against your installed fastembed
        )
    return _clip_model


def warm_up() -> None:
    """Call once at service startup so the first routed file isn't the one
    that pays the model-load cost."""
    _get_ocr_engine()
    _get_clip_model()


# ---------------------------------------------------------------------------
# Image / PDF helpers
# ---------------------------------------------------------------------------


def _load_image_as_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im)  # RGB
    return arr[:, :, ::-1].copy()  # RapidOCR/OpenCV expect BGR


def _rasterize_page(page: "fitz.Page", dpi: int = RASTER_DPI) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:  # RGBA -> drop alpha
        img = img[:, :, :3]
    return img[:, :, ::-1].copy()  # PyMuPDF gives RGB -> BGR


def _has_text_layer(doc: "fitz.Document") -> bool:
    """Defensive re-check in case a file with a real text layer slips through
    the watcher's routing. Cheap relative to OCR, so we always run it."""
    for page in doc:
        if len(page.get_text("text").strip()) >= TEXT_LAYER_MIN_CHARS:
            return True
    return False


def _run_ocr(img_bgr: np.ndarray) -> str:
    engine = _get_ocr_engine()
    result, _elapse = engine(img_bgr)
    if not result:
        return ""
    lines = [item[1] for item in result if item[1]]
    return "\n".join(lines)


def _compute_clip_vector_from_path(path: str) -> list:
    model = _get_clip_model()
    vector = next(model.embed([path]))
    return vector.tolist()


def _compute_clip_vector_from_array(img_bgr: np.ndarray) -> list:
    """fastembed's ImageEmbedding wants a file path, not a raw array, so we
    spill the rasterized page to a temp JPEG, embed it, then clean up."""
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        Image.fromarray(img_bgr[:, :, ::-1]).save(tmp_path, format="JPEG", quality=90)
        return _compute_clip_vector_from_path(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Per-type processing
# ---------------------------------------------------------------------------


def _process_image(path: Path) -> dict:
    img_bgr = _load_image_as_bgr(path)
    ocr_text = _run_ocr(img_bgr)
    vector = _compute_clip_vector_from_path(str(path))
    del img_bgr
    return {
        "file_path": str(path),
        "ocr_text": ocr_text,
        "image_vector": vector,
        "is_scanned": False,
    }


def _process_pdf(path: Path) -> dict:
    doc = fitz.open(str(path))
    try:
        if _has_text_layer(doc):
            # Shouldn't normally reach us per the routing contract, but don't
            # silently botch it if it does: use the real text, skip OCR.
            text = "\n".join(page.get_text("text") for page in doc).strip()
            first_page_img = _rasterize_page(doc[0])
            vector = _compute_clip_vector_from_array(first_page_img)
            del first_page_img
            return {
                "file_path": str(path),
                "ocr_text": text,
                "image_vector": vector,
                "is_scanned": False,
            }

        page_texts = []
        first_page_img = None
        for i, page in enumerate(doc):
            page_img = _rasterize_page(page)
            if i == 0:
                first_page_img = page_img.copy()
            page_text = _run_ocr(page_img)
            if page_text:
                page_texts.append(page_text)
            del page_img

        ocr_text = "\n\n".join(page_texts)
        # Design choice: the image_vector represents page 1 only, not an
        # average across pages — multi-page averaging tends to blur CLIP
        # similarity for docs with mixed content. Easy to change if your
        # search results say otherwise.
        vector = _compute_clip_vector_from_array(first_page_img) if first_page_img is not None else None
        del first_page_img

        return {
            "file_path": str(path),
            "ocr_text": ocr_text,
            "image_vector": vector,
            "is_scanned": True,
        }
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def process_media_file(file_path: str) -> dict:
    """Process one routed file and return its OCR text + CLIP vector.

    Returns:
        {
            "file_path": str,
            "ocr_text": str,
            "image_vector": list[float] | None,
            "is_scanned": bool,
        }
    """
    global _items_processed

    path = Path(file_path)
    ext = path.suffix.lower()

    try:
        if ext in PDF_EXTENSIONS:
            result = _process_pdf(path)
        elif ext in IMAGE_EXTENSIONS:
            result = _process_image(path)
        else:
            logger.warning("Unsupported file type routed to vision_service: %s", file_path)
            result = {
                "file_path": str(path),
                "ocr_text": "",
                "image_vector": None,
                "is_scanned": False,
            }
    except Exception:
        logger.exception("vision_service failed on %s", file_path)
        result = {
            "file_path": str(path),
            "ocr_text": "",
            "image_vector": None,
            "is_scanned": False,
        }

    _items_processed += 1
    if _items_processed % GC_EVERY_N_ITEMS == 0:
        gc.collect()
        logger.debug("gc.collect() after %d items", _items_processed)

    return result


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python vision_service.py <file_path> [file_path ...]")
        sys.exit(1)

    warm_up()
    for f in sys.argv[1:]:
        out = process_media_file(f)
        print(f"{out['file_path']}: is_scanned={out['is_scanned']}, "
              f"ocr_chars={len(out['ocr_text'])}, "
              f"vector_dim={len(out['image_vector']) if out['image_vector'] else None}")
