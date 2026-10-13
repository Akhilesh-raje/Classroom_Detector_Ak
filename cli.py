# -*- coding: utf-8 -*-
"""
SmartClass AI — Master Unified CLI
==================================
Unified entrypoint for webcam analysis, test pipeline, structure AI, backend server, and frontend.
"""
import sys
import os
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

def run_webcam(args):
    cmd = [sys.executable, os.path.join(HERE, "run_webcam.py")]
    subprocess.run(cmd)

def run_test_classroom(args):
    cmd = [sys.executable, os.path.join(HERE, "test_real_classroom.py")]
    if args.structure_ai:
        cmd.append("--structure-ai")
    if args.video:
        cmd.extend(["--video", args.video])
    subprocess.run(cmd)

def run_backend(args):
    print(f"Starting SmartClass FastAPI backend on port {args.port}...")
    import uvicorn
    sys.path.insert(0, os.path.join(HERE, "backend"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=args.port, reload=args.reload)

def run_frontend(args):
    print("Starting Vite frontend development server...")
    subprocess.run(["npm", "run", "dev"], shell=True, cwd=HERE)

def run_structure_ai(args):
    cmd = [sys.executable, os.path.join(HERE, "test_real_classroom.py"), "--structure-ai"]
    if args.video:
        cmd.extend(["--video", args.video])
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="SmartClass AI Unified Controller")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Webcam
    p_webcam = subparsers.add_parser("webcam", help="Run real-time webcam analysis")
    p_webcam.set_defaults(func=run_webcam)

    # Test Classroom
    p_test = subparsers.add_parser("test-classroom", help="Run test real classroom runner")
    p_test.add_argument("--structure-ai", action="store_true", default=True, help="Enable Structure AI pipeline")
    p_test.add_argument("--video", type=str, default="", help="Path to input video")
    p_test.set_defaults(func=run_test_classroom)

    # Structure AI directly
    p_struct = subparsers.add_parser("structure-ai", help="Run full Structure AI Comprehensive Exporter")
    p_struct.add_argument("--video", type=str, default="", help="Path to input video")
    p_struct.set_defaults(func=run_structure_ai)

    # Backend
    p_backend = subparsers.add_parser("backend", help="Run FastAPI backend server")
    p_backend.add_argument("--port", type=int, default=8000, help="Port to listen on (default 8000)")
    p_backend.add_argument("--reload", action="store_true", default=True, help="Enable auto-reload")
    p_backend.set_defaults(func=run_backend)

    # Frontend
    p_frontend = subparsers.add_parser("frontend", help="Run Vite dev server")
    p_frontend.set_defaults(func=run_frontend)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)

if __name__ == "__main__":
    main()
