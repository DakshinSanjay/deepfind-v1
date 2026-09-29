"""
download_models.py — one-shot setup script to fetch every model DeepFind's
Vision & OCR Engine (and the paired text-embedding side) needs.

None of these are committed to git — they're large binaries, and the
target machine runs fully offline. Run this once on a machine WITH
internet access, then copy the whole repo (models/ folder included) to
the offline machine.

Usage:
    python scripts/download_models.py
"""

import os
import urllib.request
from pathlib import Path

OCR_DIR = Path(os.environ.get("DEEPFIND_OCR_MODELS_DIR", "./models/ocr_en"))
CLIP_CACHE_DIR = Path(os.environ.get("DEEPFIND_CLIP_CACHE_DIR", "./models/clip_cache"))

# English-only RapidOCR trio — NOT the default multilingual bundle rapidocr
# pulls in on its own.
OCR_FILES = {
    "en_PP-OCRv3_det_infer.onnx": (
        "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/en_PP-OCRv3_det_infer.onnx"
    ),
    "en_PP-OCRv3_rec_infer.onnx": (
        "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx"
    ),
    "en_dict.txt": (
        "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/en_dict.txt"
    ),
}


def download_ocr_models() -> None:
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in OCR_FILES.items():
        dest = OCR_DIR / filename
        if dest.exists():
            print(f"[skip] {dest} already exists")
            continue
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest}")


def download_clip_models() -> None:
    # Imported here so this script's OCR-only half still works even if
    # fastembed isn't installed yet on whoever's running it.
    from fastembed import ImageEmbedding, TextEmbedding

    print("Downloading Qdrant/clip-ViT-B-32-vision (image side, used by vision_service.py) ...")
    ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision", cache_dir=str(CLIP_CACHE_DIR))

    print("Downloading Qdrant/clip-ViT-B-32-text (text side, used by the search/query path) ...")
    TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text", cache_dir=str(CLIP_CACHE_DIR))


if __name__ == "__main__":
    download_ocr_models()
    download_clip_models()
    print("\nDone. Copy the models/ folder alongside the repo onto the offline machine.")
