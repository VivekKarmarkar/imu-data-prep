#!/usr/bin/env python3
"""
Regenerate all plots in the cumulative dataset with correct randomized trial indices.
Reads from the cumulative CSVs (already copied and renamed), generates fresh plots.
Does NOT touch any other dataset outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CUMULATIVE_DIR = Path(__file__).parent / "cumulative_output"
INPUT_DIR = CUMULATIVE_DIR / "input_data"
REF_DIR = CUMULATIVE_DIR / "reference_trials"
INPUT_PLOT_DIR = INPUT_DIR / "plots"
REF_PLOT_DIR = REF_DIR / "plots"


def plot_trial_gyro(trial_idx):
    data_path = INPUT_DIR / f"trial_{trial_idx:03d}.csv"
    meta_path = INPUT_DIR / f"trial_{trial_idx:03d}_meta.csv"
    if not data_path.exists():
        return

    seg = pd.read_csv(data_path)
    meta = pd.read_csv(meta_path)

    t = seg["time"].values
    move_start_t = float(meta["move_start_time"].iloc[0])
    move_end_t = float(meta["move_end_time"].iloc[0])
    t_init = float(meta["t_init"].iloc[0])
    t_final = float(meta["t_final"].iloc[0])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, seg["Gyr_X"].values, label="Gyr_X", alpha=0.8)
    ax.plot(t, seg["Gyr_Y"].values, label="Gyr_Y", alpha=0.8)
    ax.plot(t, seg["Gyr_Z"].values, label="Gyr_Z", alpha=0.8)
    ax.axvline(move_start_t, color="red", linestyle="--", alpha=0.7, label="Move start")
    ax.axvline(move_end_t, color="green", linestyle="--", alpha=0.7, label="Move end")
    ax.axvline(t_init, color="red", linestyle=":", alpha=0.5, label=f"q_init @ {t_init:.3f}s")
    ax.axvline(t_final, color="green", linestyle=":", alpha=0.5, label=f"q_final @ {t_final:.3f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_title(f"Trial {trial_idx} — Gyroscope")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(INPUT_PLOT_DIR / f"trial_{trial_idx:03d}_gyro.png", dpi=150)
    plt.close(fig)


def plot_trial_ref(trial_idx):
    ref_path = REF_DIR / f"trial_{trial_idx:03d}_ref.csv"
    meta_path = INPUT_DIR / f"trial_{trial_idx:03d}_meta.csv"
    if not ref_path.exists():
        return

    ref_seg = pd.read_csv(ref_path)
    meta = pd.read_csv(meta_path)
    move_start_t = float(meta["move_start_time"].iloc[0])
    move_end_t = float(meta["move_end_time"].iloc[0])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(ref_seg["time"], ref_seg["qw"], label="qw", linewidth=1.2)
    ax.plot(ref_seg["time"], ref_seg["qx"], label="qx", linewidth=1.2)
    ax.plot(ref_seg["time"], ref_seg["qy"], label="qy", linewidth=1.2)
    ax.plot(ref_seg["time"], ref_seg["qz"], label="qz", linewidth=1.2)
    ax.axvline(move_start_t, color="red", linestyle="--", alpha=0.6, label="Move start")
    ax.axvline(move_end_t, color="green", linestyle="--", alpha=0.6, label="Move end")
    ax.set_ylabel("Quaternion components")
    ax.set_title(f"Trial {trial_idx} — Reference quaternion")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    qnorm = np.sqrt(ref_seg["qw"]**2 + ref_seg["qx"]**2 + ref_seg["qy"]**2 + ref_seg["qz"]**2)
    ax2.plot(ref_seg["time"], qnorm, color="black", linewidth=1.0)
    ax2.axvline(move_start_t, color="red", linestyle="--", alpha=0.6)
    ax2.axvline(move_end_t, color="green", linestyle="--", alpha=0.6)
    ax2.set_ylabel("Quaternion norm")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(0.999, 1.001)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(REF_PLOT_DIR / f"trial_{trial_idx:03d}_ref.png", dpi=150)
    plt.close(fig)


def main():
    INPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    REF_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old copied plots
    for old_plot in INPUT_PLOT_DIR.glob("*.png"):
        old_plot.unlink()
    for old_plot in REF_PLOT_DIR.glob("*.png"):
        old_plot.unlink()

    total = 272
    print(f"Regenerating all plots for {total} cumulative trials...")

    for i in range(total):
        plot_trial_gyro(i)
        plot_trial_ref(i)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{total} done")

    print(f"\nDone. All {total} gyro + reference plots regenerated with correct indices.")


if __name__ == "__main__":
    main()
