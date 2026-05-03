#!/usr/bin/env python3
"""
Strip all columns from meta CSVs except q_init and q_final quaternion components.
Retains only: q_init_w, q_init_x, q_init_y, q_init_z, q_final_w, q_final_x, q_final_y, q_final_z.

Does NOT modify trial CSVs, reference CSVs, plots, or any other files.
"""

import pandas as pd
from pathlib import Path

KEEP_COLUMNS = [
    "q_init_w", "q_init_x", "q_init_y", "q_init_z",
    "q_final_w", "q_final_x", "q_final_y", "q_final_z",
]

OUTPUT_DIRS = [
    Path(__file__).parent / "movella_output" / "input_data",
    Path(__file__).parent / "sassari_output" / "input_data",
    Path(__file__).parent / "seel_output" / "input_data",
    Path(__file__).parent / "cumulative_output" / "input_data",
]


def strip_meta(csv_path: Path):
    df = pd.read_csv(csv_path)
    if not all(col in df.columns for col in KEEP_COLUMNS):
        print(f"  SKIP (missing columns): {csv_path.name}")
        return False
    df[KEEP_COLUMNS].to_csv(csv_path, index=False)
    return True


def main():
    total = 0
    for input_dir in OUTPUT_DIRS:
        if not input_dir.exists():
            print(f"Skipping (not found): {input_dir}")
            continue

        meta_files = sorted(input_dir.glob("trial_*_meta.csv"))

        print(f"\n{input_dir.parent.name}/{input_dir.name}: {len(meta_files)} meta files")
        for f in meta_files:
            if strip_meta(f):
                total += 1

    print(f"\nDone. Stripped extra columns from {total} meta CSVs.")
    print("Columns retained: q_init_w/x/y/z, q_final_w/x/y/z")


if __name__ == "__main__":
    main()
