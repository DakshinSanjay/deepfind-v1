import time
import ingestion_service as isvc
import vision_service as vision
import search_engine as appu

def run_tier2_worker(poll_interval: float = 1.0):
    vision.warm_up()  # load OCR + CLIP once, not per file
    q = isvc.get_tier2_queue()

    while True:
        if isvc.should_pause_tier2():
            time.sleep(poll_interval)
            continue
        try:
            doc = q.get(timeout=poll_interval)
        except Exception:
            continue

        result = vision.process_media_file(doc["file_path"])
        doc["content"] = result["ocr_text"]
        doc["vector"] = result["image_vector"]
        appu.upsert(doc)
        print(f"[Tier2 -> Appu] {doc['file_name']} "
              f"(scanned={result['is_scanned']}, ocr_chars={len(result['ocr_text'])})")

if __name__ == "__main__":
    appu.init_db("./test_lancedb")
    import threading
    threading.Thread(target=run_tier2_worker, daemon=True).start()
    while True:
        time.sleep(1)
