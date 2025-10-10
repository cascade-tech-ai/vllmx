from __future__ import annotations

import argparse

import uvicorn

from .app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict demo web UI")
    parser.add_argument("--host",
                        default="127.0.0.1",
                        help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port",
                        type=int,
                        default=3000,
                        help="Port to bind (default: 3000)")
    parser.add_argument("--server-url",
                        default="http://127.0.0.1:8000",
                        help="Base URL of the vLLM server (default: http://127.0.0.1:8000)")
    parser.add_argument("--model",
                        default="predict",
                        help="Model name passed to the vLLM server (default: predict)")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Enable debug mode (disables static file caching)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(server_url=args.server_url,
                     default_model=args.model,
                     debug=args.debug)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
