#!/usr/bin/env python3
"""
Normalize all trial times to start at t=0.
Updates input_data CSVs, reference_trials CSVs, meta files, and regenerates all plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "movella_output"
INPUT_DIR = OUTPUT_DIR / "input_data"
REF_DIR = OUTPUT_DIR / "reference_trials"


def normalize_trial(trial_idx: int):
    data_path = INPUT_DIR / f"trial_{trial_idx:03d}.csv"
    meta_path = INPUT_DIR / f"trial_{trial_idx:03d}_meta.csv"
    ref_path = REF_DIR / f"trial_{trial_idx:03d}_ref.csv"

    if not data_path.exists():
        return

    data = pd.read_csv(data_path)
    meta = pd.read_csv(meta_path)
    ref = pd.read_csv(ref_path)

    t0 = data["time"].iloc[0]

    data["time"] = data["time"] - t0
    data.to_csv(data_path, index=False)

    ref["time"] = ref["time"] - t0
    ref.to_csv(ref_path, index=False)

    meta["t_init"] = meta["t_init"] - t0
    meta["t_final"] = meta["t_final"] - t0
    meta["move_start_time"] = meta["move_start_time"] - t0
    meta["move_end_time"] = meta["move_end_time"] - t0
    meta.to_csv(meta_path, index=False)

    # Regenerate gyro plot
    move_start_t = float(meta["move_start_time"].iloc[0])
    move_end_t = float(meta["move_end_time"].iloc[0])
    t_init = float(meta["t_init"].iloc[0])
    t_final = float(meta["t_final"].iloc[0])

    t = data["time"].values
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, data["Gyr_X"].values, label="Gyr_X", alpha=0.8)
    ax.plot(t, data["Gyr_Y"].values, label="Gyr_Y", alpha=0.8)
    ax.plot(t, data["Gyr_Z"].values, label="Gyr_Z", alpha=0.8)
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
    fig.savefig(INPUT_DIR / "plots" / f"trial_{trial_idx:03d}_gyro.png", dpi=150)
    plt.close(fig)

    # Regenerate reference plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(ref["time"], ref["qw"], label="qw", linewidth=1.2)
    ax.plot(ref["time"], ref["qx"], label="qx", linewidth=1.2)
    ax.plot(ref["time"], ref["qy"], label="qy", linewidth=1.2)
    ax.plot(ref["time"], ref["qz"], label="qz", linewidth=1.2)
    ax.axvline(move_start_t, color="red", linestyle="--", alpha=0.6, label="Move start")
    ax.axvline(move_end_t, color="green", linestyle="--", alpha=0.6, label="Move end")
    ax.set_ylabel("Quaternion components")
    ax.set_title(f"Trial {trial_idx} — Reference quaternion")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    qnorm = np.sqrt(ref["qw"]**2 + ref["qx"]**2 + ref["qy"]**2 + ref["qz"]**2)
    ax2.plot(ref["time"], qnorm, color="black", linewidth=1.0)
    ax2.axvline(move_start_t, color="red", linestyle="--", alpha=0.6)
    ax2.axvline(move_end_t, color="green", linestyle="--", alpha=0.6)
    ax2.set_ylabel("Quaternion norm")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(0.999, 1.001)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(REF_DIR / "plots" / f"trial_{trial_idx:03d}_ref.png", dpi=150)
    plt.close(fig)

    print(f"  Trial {trial_idx:03d}: t0={t0:.3f}s subtracted, new range [0, {data['time'].iloc[-1]:.3f}]s")


def main():
    print("Normalizing all trial times to start at t=0...")
    for i in range(64):
        normalize_trial(i)
    print("\nDone — all trials now start at t=0.")


if __name__ == "__main__":
    main()
