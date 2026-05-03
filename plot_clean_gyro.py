#!/usr/bin/env python3
"""
Plot raw gyroscope data from clean trial CSVs.
Reads from *_clean/input_data/, saves plots to *_clean_plots/.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent

DATASETS = ["movella", "sassari", "seel", "cumulative"]


def plot_trial(csv_path: Path, out_path: Path, trial_idx: str):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["time"], df["Gyr_X"], label="Gyr_X", alpha=0.8)
    ax.plot(df["time"], df["Gyr_Y"], label="Gyr_Y", alpha=0.8)
    ax.plot(df["time"], df["Gyr_Z"], label="Gyr_Z", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_title(f"Trial {trial_idx} — Gyroscope")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_dataset(name):
    input_dir = BASE / f"{name}_clean" / "input_data"
    plot_dir = BASE / f"{name}_clean_plots"

    if not input_dir.exists():
        print(f"Skipping (not found): {input_dir}")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    trial_files = sorted(f for f in input_dir.glob("trial_*.csv") if "_meta" not in f.name)
    print(f"\n{name}_clean: {len(trial_files)} trials")

    for i, f in enumerate(trial_files):
        idx = f.stem.replace("trial_", "")
        plot_trial(f, plot_dir / f"trial_{idx}_gyro.png", idx)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(trial_files)} done")

    print(f"  {len(trial_files)}/{len(trial_files)} done → {plot_dir.name}/")


def main():
    for name in DATASETS:
        process_dataset(name)
    print("\nDone.")


if __name__ == "__main__":
    main()
