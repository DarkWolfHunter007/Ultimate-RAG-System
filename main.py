import uvicorn
import os
import sys
import argparse
import logging

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultimate-RAG-System Server")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable verbose DEBUG logging for all operations")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        )
        print("⚡ [DEBUG MODE ENABLED] Full operational logging active (DEBUG level).")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s"
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)

    print(f"🚀 [Starting] Ultimate-RAG-System Server on http://{args.host}:{args.port} ...")
    uvicorn.run(
        "ui.server:app",
        host=args.host,
        port=args.port,
        reload=True,
        reload_dirs=["core", "ingestion", "retrieval", "analytics", "ui"],
        reload_excludes=["data/*", "data/**", "*.sqlite3*", "*journal*", "*.json"],
        log_level="debug" if args.debug else "info"
    )

