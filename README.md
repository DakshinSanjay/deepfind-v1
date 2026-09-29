# Offline Semantic File Search — Capstone

Local, offline, CPU-only desktop app for hybrid keyword + AI semantic search
over your files. Built with Tauri + React + FastAPI + LanceDB.

## Team & modules

| Owner    | File                     | Role                                         |
|----------|--------------------------|-----------------------------------------------|
| HP       | `ingestion_service.py`   | File watcher, system throttle, text extraction |
| Aaling   | `vision_service.py`      | OCR + CLIP image embeddings (ONNX, no PyTorch) |
| Appu     | `search_engine.py`       | LanceDB storage + hybrid search (RRF)        |
| Karthik  | `backend/main.py` + `frontend/` | FastAPI gateway, Tauri shell, React UI |

## Setup

```bash
pip install -r requirements.txt
```

## Integration contract

All ingestion payloads (from HP or Aaling) pushed into Appu's `upsert()`
must match this shape:

```python
{
    "file_path": str,
    "file_name": str,
    "file_type": str,
    "content": str,
    "vector": list[float] | None,
    "modified_time": float,
    "needs_ocr": bool,
}
```

Deletes are pushed as:
```python
{"file_path": str, "deleted": True}
```

Search results (Appu -> Karthik) look like:
```python
{
    "file_name": str,
    "file_path": str,
    "file_type": str,
    "snippet": str,
    "score": float,
    "match_type": "semantic" | "keyword" | "hybrid",
}
```

## Testing integration locally

See `test_integration.py` — wires HP's watcher directly into Appu's
`search_engine.py` so you can test end-to-end without Karthik's FastAPI/Tauri
layer running yet.

## Branching

- `main` should always be in a working state.
- Work in a personal branch (`hp-ingestion`, `appu-search`, etc.), open a PR
  to merge into `main`.
- Merge HP + Appu's integration once `test_integration.py` passes, so
  Karthik always has a working `main` to build FastAPI against.
