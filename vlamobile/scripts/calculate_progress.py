#!/usr/bin/env python
"""Add per-episode temporal progress to a LeRobot dataset.

For each frame, ``progress = frame_index / (T - 1)`` where ``T`` is the
episode length, so the first frame is 0 and the last frame is 1. Single-frame
episodes are assigned ``progress = 1``.

Walks ``<dataset_root>/data/chunk-xxx/*.parquet``, overwrites each file with a
``progress`` column, and registers the feature in ``meta/info.json`` if needed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROGRESS_FEATURE = {
    "dtype": "float64",
    "shape": [1],
    "names": None,
}


def find_parquet_files(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    files = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found under {data_dir}/chunk-*/. "
            "Expected LeRobot layout: data/chunk-xxx/*.parquet"
        )
    return files


def collect_episode_lengths(parquet_files: list[Path]) -> dict[int, int]:
    lengths: Counter[int] = Counter()
    for path in parquet_files:
        episode_index = pd.read_parquet(path, columns=["episode_index"])["episode_index"]
        lengths.update(int(ep) for ep in episode_index.to_numpy())
    return dict(lengths)


def compute_progress(frame_index: pd.Series, episode_index: pd.Series, lengths: dict[int, int]) -> pd.Series:
    episode_length = episode_index.map(lengths).astype(np.int64)
    denom = (episode_length - 1).clip(lower=1)
    progress = frame_index.astype(np.float64) / denom
    progress = progress.where(episode_length > 1, 1.0)
    return progress.astype(np.float64)


def update_info_json(dataset_root: Path) -> bool:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        print(f"No {info_path} found; skipping feature registration.")
        return False

    info = json.loads(info_path.read_text())
    features = info.setdefault("features", {})
    if features.get("progress") == PROGRESS_FEATURE:
        return False

    features["progress"] = PROGRESS_FEATURE
    info_path.write_text(json.dumps(info, indent=4) + "\n")
    return True


def calculate_progress(dataset_root: Path) -> None:
    dataset_root = dataset_root.expanduser().resolve()
    parquet_files = find_parquet_files(dataset_root)
    print(f"Found {len(parquet_files)} parquet file(s) under {dataset_root / 'data'}")

    lengths = collect_episode_lengths(parquet_files)
    print(f"Computed lengths for {len(lengths)} episode(s)")

    for path in parquet_files:
        df = pd.read_parquet(path)
        if "episode_index" not in df.columns or "frame_index" not in df.columns:
            raise ValueError(f"{path} is missing episode_index or frame_index")

        df["progress"] = compute_progress(df["frame_index"], df["episode_index"], lengths)
        df.to_parquet(path, index=False)
        print(f"Wrote progress to {path} ({len(df)} frames, {df['episode_index'].nunique()} episodes)")

    if update_info_json(dataset_root):
        print(f"Registered progress feature in {dataset_root / 'meta' / 'info.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add per-episode progress (0→1) to LeRobot parquet files."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset root containing data/chunk-xxx/*.parquet",
    )
    args = parser.parse_args()
    calculate_progress(args.dataset_root)


if __name__ == "__main__":
    main()
