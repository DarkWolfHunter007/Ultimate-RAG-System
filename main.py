import uvicorn
import os
import sys
import argparse
import logging
import logging.config

# Ensure project root is in path for absolute imports
sys.path.insert(0, os.path.dirname(__file__))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ultimate-RAG-System Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable verbose DEBUG logging for all operations (including third-party libraries)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to listen on",
    )
    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """Configure structured application logging via dictConfig.

    In DEBUG mode:  All loggers emit DEBUG-level output, including third-party
                    libraries such as httpx and chromadb — essential for tracing.
    In INFO mode:   Application loggers emit INFO+; noisy third-party loggers
                    are clamped to WARNING to reduce console noise.
    """
    log_level = "DEBUG" if debug else "INFO"
    third_party_level = "DEBUG" if debug else "WARNING"

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {
                "format": "%(asctime)s | %(levelname)-8s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "detailed" if debug else "simple",
            },
        },
        "root": {"level": log_level, "handlers": ["console"]},
        # Suppress noisy third-party loggers in production; show all in debug.
        "loggers": {
            "httpx":         {"level": third_party_level, "propagate": True},
            "httpcore":      {"level": third_party_level, "propagate": True},
            "chromadb":      {"level": third_party_level, "propagate": True},
            "uvicorn.access":{"level": third_party_level, "propagate": True},
        },
    })

    if debug:
        print("⚡ [DEBUG MODE ENABLED] Full operational logging active (DEBUG level).")


def run_server(args: argparse.Namespace) -> None:
    """Orchestrate startup: logging → server launch."""
    setup_logging(debug=args.debug)

    print(f"🚀 [Starting] Ultimate-RAG-System Server on http://{args.host}:{args.port} ...")

    uvicorn.run(
        "ui.server:app",
        host=args.host,
        port=args.port,
        reload=True,
        reload_dirs=["core", "ingestion", "retrieval", "analytics", "ui"],
        # Prevent reloads triggered by data writes or SQLite journal files.
        reload_excludes=["data/*", "data/**", "*.sqlite3*", "*journal*", "*.json"],
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    run_server(parse_args())
