#!/usr/bin/env python3
"""
Prepare movement trials from the Movella IMU dataset.

For each IMU sensor file:
1. Detect stationary vs moving segments from gyroscope data
2. Extract each movement trial with padding (stationary→moving→stationary)
3. Align trial boundaries to timestamps present in both IMU and reference data
4. Extract boundary quaternions from reference
5. Save each trial as a separate file
6. Generate verification plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d
import os
import sys

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "movella_dataset"
OUTPUT_DIR = Path(__file__).parent / "movella_output"

GYRO_THRESHOLD = 0.05       # rad/s — magnitude below this is "stationary"
SMOOTHING_WINDOW = 201      # samples for smoothing gyro magnitude (odd number)
MIN_MOVE_DURATION = 100     # minimum samples for a valid movement segment
MIN_STATIC_DURATION = 50    # minimum samples for a valid static segment
PADDING_SAMPLES = 200       # static samples to include on each side of movement

IMU_TICK_RATE = 10_000      # SampleTimeFine ticks per second


# ─── Data Loading ────────────────────────────────────────────────────────────

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


def load_imu(path: Path) -> pd.DataFrame:
    with open(path) as f:
        header_lines = 0
        for line in f:
            if line.startswith("//"):
                header_lines += 1
            else:
                col_line = line.strip()
                break

    df = pd.read_csv(path, skiprows=header_lines)
    df.columns = df.columns.str.strip()

    first_tick = df["SampleTimeFine"].iloc[0]
    df["time"] = (df["SampleTimeFine"] - first_tick) / IMU_TICK_RATE
    return df


# ─── Stationary Detection ───────────────────────────────────────────────────

def detect_segments(gyr_x: np.ndarray, gyr_y: np.ndarray, gyr_z: np.ndarray):
    """
    Returns array of booleans: True where the body is moving.
    Uses smoothed gyroscope magnitude vs threshold.
    """
    mag = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)
    smoothed = uniform_filter1d(mag, size=SMOOTHING_WINDOW)
    is_moving = smoothed > GYRO_THRESHOLD
    return is_moving, smoothed


def extract_trials(is_moving: np.ndarray, n_samples: int):
    """
    Find movement trials (contiguous moving segments) and add padding.
    Returns list of (start_idx, end_idx) with padding included.
    """
    diff = np.diff(is_moving.astype(int))
    starts = np.where(diff == 1)[0] + 1   # transition from static to moving
    ends = np.where(diff == -1)[0] + 1     # transition from moving to static

    if is_moving[0]:
        starts = np.concatenate([[0], starts])
    if is_moving[-1]:
        ends = np.concatenate([ends, [n_samples]])

    trials = []
    for s, e in zip(starts, ends):
        duration = e - s
        if duration < MIN_MOVE_DURATION:
            continue

        padded_start = max(0, s - PADDING_SAMPLES)
        padded_end = min(n_samples, e + PADDING_SAMPLES)
        trials.append({
            "move_start": s,
            "move_end": e,
            "padded_start": padded_start,
            "padded_end": padded_end,
        })

    return trials


# ─── Timestamp Alignment ────────────────────────────────────────────────────

def find_aligned_time(imu_time: float, ref_times: np.ndarray, imu_times: np.ndarray,
                      direction: str = "inward") -> float:
    """
    Find the nearest timestamp that exists in both the reference and IMU data.
    direction='inward' means search inward (right for start, left for end)
    to ensure the aligned time is within the trial.
    """
    ref_set = set(np.round(ref_times, 6))

    imu_t_rounded = np.round(imu_times, 6)

    closest_ref_idx = np.argmin(np.abs(ref_times - imu_time))
    closest_ref_time = ref_times[closest_ref_idx]

    if np.round(closest_ref_time, 6) in set(np.round(imu_times, 6)):
        return closest_ref_time

    if direction == "inward":
        for offset in range(1, 20):
            for candidate_idx in [closest_ref_idx + offset, closest_ref_idx - offset]:
                if 0 <= candidate_idx < len(ref_times):
                    t = ref_times[candidate_idx]
                    if np.round(t, 6) in set(np.round(imu_times, 6)):
                        return t

    return closest_ref_time


# ─── Output ─────────────────────────────────────────────────────────────────

def save_trial(trial_data: dict, sensor_name: str, trial_idx: int, output_dir: Path):
    trial_dir = output_dir / sensor_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    fname = trial_dir / f"trial_{trial_idx:03d}.csv"
    df = trial_data["gyro_segment"].copy()
    df.to_csv(fname, index=False)

    meta_fname = trial_dir / f"trial_{trial_idx:03d}_meta.csv"
    meta = pd.DataFrame([{
        "trial_index": trial_idx,
        "q_init_w": trial_data["q_init"][0],
        "q_init_x": trial_data["q_init"][1],
        "q_init_y": trial_data["q_init"][2],
        "q_init_z": trial_data["q_init"][3],
        "q_final_w": trial_data["q_final"][0],
        "q_final_x": trial_data["q_final"][1],
        "q_final_y": trial_data["q_final"][2],
        "q_final_z": trial_data["q_final"][3],
        "t_init": trial_data["t_init"],
        "t_final": trial_data["t_final"],
        "move_start_time": trial_data["move_start_time"],
        "move_end_time": trial_data["move_end_time"],
    }])
    meta.to_csv(meta_fname, index=False)


def plot_trial(trial_data: dict, sensor_name: str, trial_idx: int, output_dir: Path):
    trial_dir = output_dir / sensor_name / "plots"
    trial_dir.mkdir(parents=True, exist_ok=True)

    seg = trial_data["gyro_segment"]
    t = seg["time"].values
    move_start_t = trial_data["move_start_time"]
    move_end_t = trial_data["move_end_time"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, seg["Gyr_X"].values, label="Gyr_X", alpha=0.8)
    ax.plot(t, seg["Gyr_Y"].values, label="Gyr_Y", alpha=0.8)
    ax.plot(t, seg["Gyr_Z"].values, label="Gyr_Z", alpha=0.8)
    ax.axvline(move_start_t, color="red", linestyle="--", alpha=0.7, label="Move start")
    ax.axvline(move_end_t, color="green", linestyle="--", alpha=0.7, label="Move end")
    ax.axvline(trial_data["t_init"], color="red", linestyle=":", alpha=0.5, label=f"q_init @ {trial_data['t_init']:.3f}s")
    ax.axvline(trial_data["t_final"], color="green", linestyle=":", alpha=0.5, label=f"q_final @ {trial_data['t_final']:.3f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_title(f"{sensor_name} — Trial {trial_idx}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(trial_dir / f"trial_{trial_idx:03d}_gyro.png", dpi=150)
    plt.close(fig)


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def process_sensor(imu_path: Path, ref_df: pd.DataFrame, output_dir: Path):
    sensor_name = imu_path.stem
    print(f"\n{'='*60}")
    print(f"Processing sensor: {sensor_name}")
    print(f"{'='*60}")

    imu_df = load_imu(imu_path)
    print(f"  IMU samples: {len(imu_df)}, time range: {imu_df['time'].iloc[0]:.3f} – {imu_df['time'].iloc[-1]:.3f} s")

    gyr_x = imu_df["Gyr_X"].values
    gyr_y = imu_df["Gyr_Y"].values
    gyr_z = imu_df["Gyr_Z"].values

    is_moving, smoothed_mag = detect_segments(gyr_x, gyr_y, gyr_z)
    trials = extract_trials(is_moving, len(imu_df))
    print(f"  Detected {len(trials)} movement trials")

    # Overview plot: full gyro magnitude + detected segments
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(imu_df["time"].values, smoothed_mag, color="steelblue", linewidth=0.5)
    ax.axhline(GYRO_THRESHOLD, color="red", linestyle="--", alpha=0.5, label=f"Threshold = {GYRO_THRESHOLD}")
    for i, tr in enumerate(trials):
        t_s = imu_df["time"].iloc[tr["padded_start"]]
        t_e = imu_df["time"].iloc[min(tr["padded_end"], len(imu_df)-1)]
        ax.axvspan(t_s, t_e, alpha=0.15, color="orange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Smoothed gyro magnitude (rad/s)")
    ax.set_title(f"{sensor_name} — Movement detection overview")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    overview_dir = output_dir / sensor_name / "plots"
    overview_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(overview_dir / "overview_detection.png", dpi=150)
    plt.close(fig)

    ref_times = ref_df["time"].values
    imu_times = imu_df["time"].values

    for i, trial in enumerate(trials):
        seg = imu_df.iloc[trial["padded_start"]:trial["padded_end"]].copy()
        seg = seg[["time", "Gyr_X", "Gyr_Y", "Gyr_Z", "Acc_X", "Acc_Y", "Acc_Z"]].reset_index(drop=True)

        move_start_time = imu_df["time"].iloc[trial["move_start"]]
        move_end_time = imu_df["time"].iloc[min(trial["move_end"], len(imu_df)-1)]

        t_init = find_aligned_time(seg["time"].iloc[0], ref_times, imu_times)
        t_final = find_aligned_time(seg["time"].iloc[-1], ref_times, imu_times)

        ref_init_idx = np.argmin(np.abs(ref_times - t_init))
        ref_final_idx = np.argmin(np.abs(ref_times - t_final))

        q_init = ref_df.iloc[ref_init_idx][["qw", "qx", "qy", "qz"]].values.astype(float)
        q_final = ref_df.iloc[ref_final_idx][["qw", "qx", "qy", "qz"]].values.astype(float)

        trial_data = {
            "gyro_segment": seg,
            "q_init": q_init,
            "q_final": q_final,
            "t_init": t_init,
            "t_final": t_final,
            "move_start_time": move_start_time,
            "move_end_time": move_end_time,
        }

        save_trial(trial_data, sensor_name, i, output_dir)
        plot_trial(trial_data, sensor_name, i, output_dir)
        print(f"  Trial {i}: t=[{seg['time'].iloc[0]:.3f}, {seg['time'].iloc[-1]:.3f}]s, "
              f"move=[{move_start_time:.3f}, {move_end_time:.3f}]s, "
              f"q_init=({q_init[0]:.4f},...), q_final=({q_final[0]:.4f},...)")

    print(f"\n  Output saved to: {output_dir / sensor_name}")
    return len(trials)


def main():
    ref_path = DATA_DIR / "reference.txt"
    ref_df = load_reference(ref_path)
    print(f"Reference loaded: {len(ref_df)} rows, time range: {ref_df['time'].iloc[0]:.3f} – {ref_df['time'].iloc[-1]:.3f} s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    imu_files = sorted(DATA_DIR.glob("MT_*.txt"))
    print(f"Found {len(imu_files)} IMU sensor files")

    total_trials = 0
    for imu_path in imu_files:
        n = process_sensor(imu_path, ref_df, OUTPUT_DIR)
        total_trials += n

    print(f"\n{'='*60}")
    print(f"Done. Total trials extracted: {total_trials}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
