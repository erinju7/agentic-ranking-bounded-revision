from __future__ import annotations

import argparse
from pathlib import Path

from retriever import ARTIFACT_DIR, DATA_PATH, DEFAULT_MIN_PROJECTS_PER_CALL, build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute the funding-call retriever artifacts.",
    )
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    parser.add_argument(
        "--min-projects-per-call",
        type=int,
        default=DEFAULT_MIN_PROJECTS_PER_CALL,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_index(
        data_path=Path(args.data_path),
        artifact_dir=Path(args.artifact_dir),
        min_projects_per_call=args.min_projects_per_call,
    )
    print("Built retriever artifacts")
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
