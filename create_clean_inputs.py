#!/usr/bin/env python3
"""
Create clean input directories with ONLY the requested columns:
  - trial CSVs: time, Gyr_X, Gyr_Y, Gyr_Z
  - meta CSVs: q_init_w, q_init_x, q_init_y, q_init_z, q_final_w, q_final_x, q_final_y, q_final_z

Reads from existing dataset outputs, writes to NEW directories.
Does NOT modify any existing files.
"""

import pandas as pd
from pathlib import Path

GYRO_COLS = ["time", "Gyr_X", "Gyr_Y", "Gyr_Z"]
META_COLS = [
    "q_init_w", "q_init_x", "q_init_y", "q_init_z",
    "q_final_w", "q_final_x", "q_final_y", "q_final_z",
]

BASE = Path(__file__).parent

SOURCES = [
    ("movella_output", "movella_clean"),
    ("sassari_output", "sassari_clean"),
    ("seel_output", "seel_clean"),
    ("cumulative_output", "cumulative_clean"),
]


def process_dataset(src_name, dst_name):
    src_dir = BASE / src_name / "input_data"
    dst_dir = BASE / dst_name / "input_data"

    if not src_dir.exists():
        print(f"Skipping (not found): {src_dir}")
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    trial_files = sorted(f for f in src_dir.glob("trial_*.csv") if "_meta" not in f.name)
    count = 0

    for trial_path in trial_files:
        stem = trial_path.stem
        meta_path = src_dir / f"{stem}_meta.csv"

        df = pd.read_csv(trial_path)
        df[GYRO_COLS].to_csv(dst_dir / trial_path.name, index=False)

        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            meta[META_COLS].to_csv(dst_dir / meta_path.name, index=False)

        count += 1

    return count


def main():
    total = 0
    for src, dst in SOURCES:
        n = process_dataset(src, dst)
        print(f"{src} → {dst}: {n} trials")
        total += n

    print(f"\nDone. {total} trials written to clean directories.")
    print("Trial columns: time, Gyr_X, Gyr_Y, Gyr_Z")
    print("Meta columns: q_init_w/x/y/z, q_final_w/x/y/z")


if __name__ == "__main__":
    main()
