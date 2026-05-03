#!/usr/bin/env python3
"""
Extract reference quaternion time series for each movement trial
identified in the Movella dataset, and generate plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "movella_dataset"
OUTPUT_DIR = Path(__file__).parent / "movella_output"
REF_OUTPUT_DIR = OUTPUT_DIR / "reference_trials"


def load_reference(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "time": "time",
        "quaternion_w": "qw",
        "quaternion_x": "qx",
        "quaternion_y": "qy",
        "quaternion_z": "qz",
    })
    return df


def main():
    ref_df = load_reference(DATA_DIR / "reference.txt")
    ref_times = ref_df["time"].values

    # Use first sensor's trials as the canonical trial boundaries
    sensor_dir = sorted((OUTPUT_DIR).glob("MT_*"))[0]
    sensor_name = sensor_dir.name
    print(f"Using trial boundaries from: {sensor_name}")

    meta_files = sorted(sensor_dir.glob("trial_*_meta.csv"))
    print(f"Found {len(meta_files)} trials")

    REF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = REF_OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for meta_path in meta_files:
        meta = pd.read_csv(meta_path)
        trial_idx = int(meta["trial_index"].iloc[0])
        t_init = float(meta["t_init"].iloc[0])
        t_final = float(meta["t_final"].iloc[0])
        move_start = float(meta["move_start_time"].iloc[0])
        move_end = float(meta["move_end_time"].iloc[0])

        mask = (ref_df["time"] >= t_init) & (ref_df["time"] <= t_final)
        ref_seg = ref_df[mask][["time", "qw", "qx", "qy", "qz"]].copy().reset_index(drop=True)

        ref_seg.to_csv(REF_OUTPUT_DIR / f"trial_{trial_idx:03d}_ref.csv", index=False)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax = axes[0]
        ax.plot(ref_seg["time"], ref_seg["qw"], label="qw", linewidth=1.2)
        ax.plot(ref_seg["time"], ref_seg["qx"], label="qx", linewidth=1.2)
        ax.plot(ref_seg["time"], ref_seg["qy"], label="qy", linewidth=1.2)
        ax.plot(ref_seg["time"], ref_seg["qz"], label="qz", linewidth=1.2)
        ax.axvline(move_start, color="red", linestyle="--", alpha=0.6, label="Move start")
        ax.axvline(move_end, color="green", linestyle="--", alpha=0.6, label="Move end")
        ax.set_ylabel("Quaternion components")
        ax.set_title(f"Trial {trial_idx} — Reference quaternion")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Quaternion norm (should be ~1.0 throughout)
        ax2 = axes[1]
        qnorm = np.sqrt(ref_seg["qw"]**2 + ref_seg["qx"]**2 + ref_seg["qy"]**2 + ref_seg["qz"]**2)
        ax2.plot(ref_seg["time"], qnorm, color="black", linewidth=1.0)
        ax2.axvline(move_start, color="red", linestyle="--", alpha=0.6)
        ax2.axvline(move_end, color="green", linestyle="--", alpha=0.6)
        ax2.set_ylabel("Quaternion norm")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylim(0.999, 1.001)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_dir / f"trial_{trial_idx:03d}_ref.png", dpi=150)
        plt.close(fig)

        print(f"  Trial {trial_idx}: t=[{t_init:.3f}, {t_final:.3f}]s, "
              f"{len(ref_seg)} ref samples, "
              f"q_start=({ref_seg['qw'].iloc[0]:.4f},...), "
              f"q_end=({ref_seg['qw'].iloc[-1]:.4f},...)")

    print(f"\nReference trials saved to: {REF_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
