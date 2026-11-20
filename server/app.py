"""Loopback-only HTTP entry point for the bundled experimental ONNX model."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from recognition.model import load_model
from recognition.service import create_app as create_service

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"


def create_app(token_file: Path | None = None, model_dir: Path = DEFAULT_MODEL_DIR,
               capacity: int = 20, workers: int = 2, cpu_threads: int = 1):
    """Create the authenticated API, with 503 readiness on invalid model assets."""
    try:
        predictor = load_model(Path(model_dir), threads=cpu_threads)
    except ValueError:
        logging.getLogger("recognition.startup").error(json.dumps({"event": "model_unavailable"}))
        predictor = None
    return create_service(token_file=token_file, predictor=predictor,
                          capacity=capacity, workers=workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, required=True,
                        help="Backend-only bearer credential file, at least 32 printable ASCII characters")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.cpu_threads <= 32:
        parser.error("Use a valid TCP port and 1–32 CPU threads.")
    import uvicorn
    uvicorn.run(create_app(args.token_file, args.model_dir, cpu_threads=args.cpu_threads),
                host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
