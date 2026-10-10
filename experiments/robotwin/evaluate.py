"""Launch RoboTwin Clean and Randomized evaluation workflows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to a TurboVLA checkpoint")
    parser.add_argument(
        "--mode",
        choices=("both", "clean", "randomized"),
        default=os.environ.get("ROBOTWIN_EVAL_MODE", "both"),
        help="Evaluation variant (default: both Clean and Randomized)",
    )
    parser.add_argument(
        "tasks",
        nargs="*",
        help="Optional RoboTwin task names; omit to evaluate all 50 tasks",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(
        os.environ.get("TURBOVLA_REPO_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()
    script = repo_root / "scripts" / "robotwin" / "evaluate.sh"
    if not script.is_file():
        raise FileNotFoundError(f"RoboTwin evaluation script not found: {script}")
    subprocess.run(
        ["bash", str(script), args.checkpoint, "--mode", args.mode, *args.tasks],
        cwd=repo_root,
        check=True,
    )


if __name__ == "__main__":
    main()
