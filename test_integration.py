"""
test_integration.py
--------------------
Shared test: wires HP's ingestion_service directly into Appu's
search_engine, with no FastAPI/Tauri layer needed.

Usage:
    python test_integration.py
    (drop files into ./test_watch_folder while it's running)
    Ctrl+C to stop and run a search
"""

import ingestion_service as isvc
import search_engine as appu

appu.init_db("./test_lancedb")


def route_to_appu(doc: dict):
    if doc.get("deleted"):
        appu.delete(doc["file_path"])
        print(f"[Appu.delete] {doc['file_path']}")
    else:
        appu.upsert(doc)
        print(f"[Appu.upsert] {doc['file_name']} ({len(doc.get('content', ''))} chars, "
              f"needs_ocr={doc.get('needs_ocr')})")


isvc.register_appu_callback(route_to_appu)
isvc.update_limits(cpu_limit=50, ram_limit=75, pause_on_battery=False)
isvc.start_watcher(["./test_watch_folder"])

print("Watching ./test_watch_folder. Drop files in, then Ctrl+C to search.")

try:
    while True:
        pass
except KeyboardInterrupt:
    isvc.stop_watcher()
    query = input("\nEnter a search query to test: ")
    results = appu.search(query)
    print("\n--- Search results ---")
    for r in results:
        print(r)
