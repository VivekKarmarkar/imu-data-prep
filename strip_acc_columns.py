#!/usr/bin/env python3
"""
Strip accelerometer columns from all trial CSVs across all dataset outputs
and the cumulative output. Reads existing trial CSVs, writes new versions
with only: time, Gyr_X, Gyr_Y, Gyr_Z.

Does NOT modify meta CSVs, reference CSVs, plots, or any other files.
"""

import pandas as pd
from pathlib import Path

GYRO_COLUMNS = ["time", "Gyr_X", "Gyr_Y", "Gyr_Z"]

OUTPUT_DIRS = [
    Path(__file__).parent / "movella_output" / "input_data",
    Path(__file__).parent / "sassari_output" / "input_data",
    Path(__file__).parent / "seel_output" / "input_data",
    Path(__file__).parent / "cumulative_output" / "input_data",
]


def strip_trial(csv_path: Path):
    df = pd.read_csv(csv_path)
    if not all(col in df.columns for col in GYRO_COLUMNS):
        print(f"  SKIP (missing gyro columns): {csv_path.name}")
        return False
    df_gyro = df[GYRO_COLUMNS]
    df_gyro.to_csv(csv_path, index=False)
    return True


def main():
    total = 0
    for input_dir in OUTPUT_DIRS:
        if not input_dir.exists():
            print(f"Skipping (not found): {input_dir}")
            continue

        trial_files = sorted(input_dir.glob("trial_*.csv"))
        trial_files = [f for f in trial_files if "_meta" not in f.name]

        print(f"\n{input_dir.parent.name}/{input_dir.name}: {len(trial_files)} trial files")
        for f in trial_files:
            if strip_trial(f):
                total += 1

    print(f"\nDone. Stripped Acc columns from {total} trial CSVs.")
    print("Columns retained: time, Gyr_X, Gyr_Y, Gyr_Z")


if __name__ == "__main__":
    main()
