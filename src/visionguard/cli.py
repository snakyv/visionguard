from __future__ import annotations

import argparse
import json
import os

import uvicorn

from visionguard.demo import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(prog="visionguard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default=os.getenv("VISIONGUARD_HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=int(os.getenv("VISIONGUARD_PORT", "8000")))

    demo = subparsers.add_parser("demo")
    demo.add_argument("--source", default="datasets/demo/demo_scene.mp4")
    demo.add_argument("--output", default="outputs/demo_annotated.mp4")
    demo.add_argument("--max-frames", type=int, default=None)

    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run("visionguard.api:app", host=args.host, port=args.port, reload=False)
    elif args.command == "demo":
        result = run_demo(args.source, args.output, args.max_frames)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
