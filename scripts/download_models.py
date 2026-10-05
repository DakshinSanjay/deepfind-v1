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
OCR_ACCURACY_DIR = Path(
    os.environ.get("DEEPFIND_OCR_ACCURACY_DIR", "./models/ocr_accuracy")
)
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

OCR_ACCURACY_FILES = {
    "ch_PP-OCRv4_det_infer.onnx": (
        "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx"
    ),
    "ch_PP-OCRv4_rec_infer.onnx": (
        "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx"
    ),
    "ch_dict.txt": (
        "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt"
    ),
}


def _download_files(directory: Path, files: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, url in files.items():
        dest = directory / filename
        if dest.exists():
            print(f"[skip] {dest} already exists")
            continue
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest}")


def download_ocr_models() -> None:
    _download_files(OCR_DIR, OCR_FILES)
    _download_files(OCR_ACCURACY_DIR, OCR_ACCURACY_FILES)


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
