#!/usr/bin/env python3
"""
Renumber trials into a single global sequence (0-63) across all 4 sensors,
and duplicate reference trials to match each sensor's renumbered trials.
"""

import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "movella_output"
REF_DIR = OUTPUT_DIR / "reference_trials"


def renumber_sensor_trials(sensor_dir: Path, offset: int):
    """Renumber trial files in a sensor directory by adding offset to indices."""
    meta_files = sorted(sensor_dir.glob("trial_*_meta.csv"))

    for meta_path in meta_files:
        old_idx = int(meta_path.stem.split("_")[1])
        new_idx = old_idx + offset

        # Rename data file
        old_data = sensor_dir / f"trial_{old_idx:03d}.csv"
        new_data = sensor_dir / f"trial_{new_idx:03d}.csv"
        old_data.rename(new_data)

        # Update and rename meta file
        meta_df = pd.read_csv(meta_path)
        meta_df["trial_index"] = new_idx
        new_meta = sensor_dir / f"trial_{new_idx:03d}_meta.csv"
        meta_df.to_csv(new_meta, index=False)
        if meta_path != new_meta:
            meta_path.unlink()

        # Rename plot
        old_plot = sensor_dir / "plots" / f"trial_{old_idx:03d}_gyro.png"
        new_plot = sensor_dir / "plots" / f"trial_{new_idx:03d}_gyro.png"
        if old_plot.exists():
            old_plot.rename(new_plot)

        print(f"  {sensor_dir.name}: trial {old_idx:03d} -> {new_idx:03d}")


def duplicate_reference_trials(offset: int, n_trials: int = 16):
    """Duplicate reference trial files with new indices starting at offset."""
    plot_dir = REF_DIR / "plots"

    for old_idx in range(n_trials):
        new_idx = old_idx + offset

        # Copy ref CSV
        old_csv = REF_DIR / f"trial_{old_idx:03d}_ref.csv"
        new_csv = REF_DIR / f"trial_{new_idx:03d}_ref.csv"
        shutil.copy2(old_csv, new_csv)

        # Copy ref plot
        old_plot = plot_dir / f"trial_{old_idx:03d}_ref.png"
        new_plot = plot_dir / f"trial_{new_idx:03d}_ref.png"
        shutil.copy2(old_plot, new_plot)

        print(f"  reference: trial {old_idx:03d} -> {new_idx:03d} (duplicated)")


def update_plot_titles(sensor_dir: Path, offset: int, n_trials: int = 16):
    """Regenerate gyro plots with updated trial index in title."""
    plot_dir = sensor_dir / "plots"

    for local_idx in range(n_trials):
        new_idx = local_idx + offset
        data_path = sensor_dir / f"trial_{new_idx:03d}.csv"
        meta_path = sensor_dir / f"trial_{new_idx:03d}_meta.csv"

        if not data_path.exists():
            continue

        seg = pd.read_csv(data_path)
        meta = pd.read_csv(meta_path)
        move_start_t = float(meta["move_start_time"].iloc[0])
        move_end_t = float(meta["move_end_time"].iloc[0])
        t_init = float(meta["t_init"].iloc[0])
        t_final = float(meta["t_final"].iloc[0])

        t = seg["time"].values
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
        ax.set_title(f"{sensor_dir.name} — Trial {new_idx}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"trial_{new_idx:03d}_gyro.png", dpi=150)
        plt.close(fig)


def update_ref_plot_titles(offset: int, n_trials: int = 16):
    """Regenerate reference plots with updated trial index in title."""
    plot_dir = REF_DIR / "plots"

    for local_idx in range(n_trials):
        new_idx = local_idx + offset
        ref_path = REF_DIR / f"trial_{new_idx:03d}_ref.csv"
        if not ref_path.exists():
            continue

        # Need the meta from any sensor for move start/end times
        sensor_dirs = sorted(OUTPUT_DIR.glob("MT_*"))
        meta_path = sensor_dirs[0] / f"trial_{local_idx:03d}_meta.csv"
        if not meta_path.exists():
            meta_path = sensor_dirs[0] / f"trial_{new_idx:03d}_meta.csv"
        meta = pd.read_csv(meta_path)
        move_start = float(meta["move_start_time"].iloc[0])
        move_end = float(meta["move_end_time"].iloc[0])

        ref_seg = pd.read_csv(ref_path)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax = axes[0]
        ax.plot(ref_seg["time"], ref_seg["qw"], label="qw", linewidth=1.2)
        ax.plot(ref_seg["time"], ref_seg["qx"], label="qx", linewidth=1.2)
        ax.plot(ref_seg["time"], ref_seg["qy"], label="qy", linewidth=1.2)
        ax.plot(ref_seg["time"], ref_seg["qz"], label="qz", linewidth=1.2)
        ax.axvline(move_start, color="red", linestyle="--", alpha=0.6, label="Move start")
        ax.axvline(move_end, color="green", linestyle="--", alpha=0.6, label="Move end")
        ax.set_ylabel("Quaternion components")
        ax.set_title(f"Trial {new_idx} — Reference quaternion")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

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
        fig.savefig(plot_dir / f"trial_{new_idx:03d}_ref.png", dpi=150)
        plt.close(fig)


def main():
    sensor_dirs = sorted(OUTPUT_DIR.glob("MT_*"))
    print(f"Found {len(sensor_dirs)} sensor directories")

    offsets = {0: 0, 1: 16, 2: 32, 3: 48}

    # Renumber sensors 1-3 (sensor 0 stays at 0-15)
    for i in [3, 2, 1]:  # reverse order to avoid collisions
        print(f"\nRenumbering sensor {i} ({sensor_dirs[i].name}), offset={offsets[i]}:")
        renumber_sensor_trials(sensor_dirs[i], offsets[i])

    print(f"\nSensor 0 ({sensor_dirs[0].name}) stays at 0-15")

    # Duplicate reference trials for sensors 1-3
    for i in [1, 2, 3]:
        print(f"\nDuplicating reference trials for sensor {i}, offset={offsets[i]}:")
        duplicate_reference_trials(offsets[i])

    # Update plot titles for renumbered sensors
    for i in [1, 2, 3]:
        print(f"\nUpdating plot titles for sensor {i}...")
        update_plot_titles(sensor_dirs[i], offsets[i])

    # Update ref plot titles for duplicated references
    for i in [1, 2, 3]:
        print(f"Updating reference plot titles for offset {offsets[i]}...")
        update_ref_plot_titles(offsets[i])

    # Summary
    print(f"\n{'='*60}")
    print("Renumbering complete:")
    for i, sd in enumerate(sensor_dirs):
        start = offsets[i]
        end = offsets[i] + 15
        print(f"  {sd.name}: trials {start:03d} – {end:03d}")
    print(f"  reference_trials: trials 000 – 063")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
