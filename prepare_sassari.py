#!/usr/bin/env python3
"""
Prepare movement trials from the Sassari IMU dataset.

Three speed conditions (slow, medium, fast), six sensors each (AP1, AP2, SH1, SH2, XS1, XS2),
with Vicon reference quaternions. Same pipeline as Movella: detect movements, extract with
padding, align to reference, generate verification plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from scipy.ndimage import uniform_filter1d

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "sassari_dataset" / "sassari_imu_dataset"
OUTPUT_DIR = Path(__file__).parent / "sassari_output"

SAMPLE_RATE = 100  # Hz (paper: all resampled to 100 Hz)

GYRO_THRESHOLD = 0.05
SMOOTHING_WINDOW = 51   # samples at 100 Hz ≈ 0.5s
MIN_MOVE_DURATION = 30  # samples at 100 Hz ≈ 0.3s
PADDING_SAMPLES = 50    # samples at 100 Hz ≈ 0.5s

SENSOR_NAMES = ["AP1", "AP2", "SH1", "SH2", "XS1", "XS2"]
SPEED_FILES = ["slow_v5.mat", "medium_v5.mat", "fast_v5.mat"]

# Sensor column layout: [0]=timestamp, [1:4]=acc, [4:7]=gyro, [7:10]=mag, [10:14]=quaternion


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_sassari_mat(path: Path):
    data = sio.loadmat(str(path))
    n_samples = data["Qs"].shape[0]
    time = np.arange(n_samples) / SAMPLE_RATE

    ref_df = pd.DataFrame({
        "time": time,
        "qw": data["Qs"][:, 0],
        "qx": data["Qs"][:, 1],
        "qy": data["Qs"][:, 2],
        "qz": data["Qs"][:, 3],
    })

    sensors = {}
    for name in SENSOR_NAMES:
        raw = data[name]
        sensors[name] = pd.DataFrame({
            "time": time,
            "Acc_X": raw[:, 1],
            "Acc_Y": raw[:, 2],
            "Acc_Z": raw[:, 3],
            "Gyr_X": raw[:, 4],
            "Gyr_Y": raw[:, 5],
            "Gyr_Z": raw[:, 6],
        })

    return ref_df, sensors


# ─── Stationary Detection ───────────────────────────────────────────────────

def detect_segments(gyr_x, gyr_y, gyr_z):
    mag = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)
    smoothed = uniform_filter1d(mag, size=SMOOTHING_WINDOW)
    is_moving = smoothed > GYRO_THRESHOLD
    return is_moving, smoothed


def extract_trials(is_moving, n_samples):
    diff = np.diff(is_moving.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if is_moving[0]:
        starts = np.concatenate([[0], starts])
    if is_moving[-1]:
        ends = np.concatenate([ends, [n_samples]])

    trials = []
    for s, e in zip(starts, ends):
        if (e - s) < MIN_MOVE_DURATION:
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


# ─── Output ─────────────────────────────────────────────────────────────────

def save_trial(seg, q_init, q_final, t_init, t_final, move_start_t, move_end_t,
               trial_idx, output_dir):
    seg.to_csv(output_dir / f"trial_{trial_idx:03d}.csv", index=False)

    meta = pd.DataFrame([{
        "trial_index": trial_idx,
        "q_init_w": q_init[0], "q_init_x": q_init[1],
        "q_init_y": q_init[2], "q_init_z": q_init[3],
        "q_final_w": q_final[0], "q_final_x": q_final[1],
        "q_final_y": q_final[2], "q_final_z": q_final[3],
        "t_init": t_init, "t_final": t_final,
        "move_start_time": move_start_t, "move_end_time": move_end_t,
    }])
    meta.to_csv(output_dir / f"trial_{trial_idx:03d}_meta.csv", index=False)


def plot_trial_gyro(seg, move_start_t, move_end_t, t_init, t_final,
                    trial_idx, plot_dir):
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
    ax.set_title(f"Trial {trial_idx} — Gyroscope")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / f"trial_{trial_idx:03d}_gyro.png", dpi=150)
    plt.close(fig)


def plot_trial_ref(ref_seg, move_start_t, move_end_t, trial_idx, plot_dir):
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
    fig.savefig(plot_dir / f"trial_{trial_idx:03d}_ref.png", dpi=150)
    plt.close(fig)


def plot_overview(sensor_df, smoothed_mag, trials, sensor_label, trial_offset, plot_dir):
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(sensor_df["time"].values, smoothed_mag, color="steelblue", linewidth=0.5)
    ax.axhline(GYRO_THRESHOLD, color="red", linestyle="--", alpha=0.5,
               label=f"Threshold = {GYRO_THRESHOLD}")
    for tr in trials:
        t_s = sensor_df["time"].iloc[tr["padded_start"]]
        t_e = sensor_df["time"].iloc[min(tr["padded_end"], len(sensor_df)-1)]
        ax.axvspan(t_s, t_e, alpha=0.15, color="orange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Smoothed gyro magnitude (rad/s)")
    ax.set_title(f"{sensor_label} — Movement detection overview")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / f"overview_{sensor_label}.png", dpi=150)
    plt.close(fig)


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def process_speed(speed_file: str, global_offset: int):
    speed_name = speed_file.replace("_v5.mat", "")
    print(f"\n{'='*60}")
    print(f"Processing speed: {speed_name}")
    print(f"{'='*60}")

    ref_df, sensors = load_sassari_mat(DATA_DIR / speed_file)
    n_samples = len(ref_df)
    print(f"  Samples: {n_samples}, duration: {n_samples/SAMPLE_RATE:.1f}s")

    input_dir = OUTPUT_DIR / "input_data"
    input_plot_dir = input_dir / "plots"
    ref_out_dir = OUTPUT_DIR / "reference_trials"
    ref_plot_dir = ref_out_dir / "plots"
    for d in [input_dir, input_plot_dir, ref_out_dir, ref_plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    current_offset = global_offset

    for sensor_idx, sensor_name in enumerate(SENSOR_NAMES):
        sensor_df = sensors[sensor_name]
        speed_sensor_label = f"{speed_name}_{sensor_name}"
        print(f"\n  Sensor: {sensor_name}")

        gyr_x = sensor_df["Gyr_X"].values
        gyr_y = sensor_df["Gyr_Y"].values
        gyr_z = sensor_df["Gyr_Z"].values

        is_moving, smoothed_mag = detect_segments(gyr_x, gyr_y, gyr_z)
        trials = extract_trials(is_moving, n_samples)
        print(f"    Detected {len(trials)} movement trials")

        plot_overview(sensor_df, smoothed_mag, trials, speed_sensor_label,
                      current_offset, input_plot_dir)

        for i, trial in enumerate(trials):
            trial_idx = current_offset + i

            seg = sensor_df.iloc[trial["padded_start"]:trial["padded_end"]].copy()
            seg = seg[["time", "Gyr_X", "Gyr_Y", "Gyr_Z", "Acc_X", "Acc_Y", "Acc_Z"]].reset_index(drop=True)

            move_start_t = sensor_df["time"].iloc[trial["move_start"]]
            move_end_t = sensor_df["time"].iloc[min(trial["move_end"], n_samples-1)]

            t_init = seg["time"].iloc[0]
            t_final = seg["time"].iloc[-1]

            # Reference quaternion at boundaries
            ref_init_idx = trial["padded_start"]
            ref_final_idx = min(trial["padded_end"] - 1, len(ref_df) - 1)
            q_init = ref_df.iloc[ref_init_idx][["qw", "qx", "qy", "qz"]].values.astype(float)
            q_final = ref_df.iloc[ref_final_idx][["qw", "qx", "qy", "qz"]].values.astype(float)

            # Reference quaternion segment
            ref_seg = ref_df.iloc[trial["padded_start"]:trial["padded_end"]].copy()
            ref_seg = ref_seg[["time", "qw", "qx", "qy", "qz"]].reset_index(drop=True)

            # Normalize time to start at 0
            t0 = seg["time"].iloc[0]
            seg["time"] = seg["time"] - t0
            ref_seg["time"] = ref_seg["time"] - t0
            move_start_t -= t0
            move_end_t -= t0
            t_init -= t0
            t_final -= t0

            save_trial(seg, q_init, q_final, t_init, t_final,
                       move_start_t, move_end_t, trial_idx, input_dir)

            ref_seg.to_csv(ref_out_dir / f"trial_{trial_idx:03d}_ref.csv", index=False)

            plot_trial_gyro(seg, move_start_t, move_end_t, t_init, t_final,
                            trial_idx, input_plot_dir)
            plot_trial_ref(ref_seg, move_start_t, move_end_t, trial_idx, ref_plot_dir)

            print(f"    Trial {trial_idx}: t=[0, {seg['time'].iloc[-1]:.3f}]s, "
                  f"move=[{move_start_t:.3f}, {move_end_t:.3f}]s")

        current_offset += len(trials)

    return current_offset


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global_offset = 0
    for speed_file in SPEED_FILES:
        global_offset = process_speed(speed_file, global_offset)

    print(f"\n{'='*60}")
    print(f"Done. Total trials extracted: {global_offset}")
    print(f"Trial range: 000 – {global_offset-1:03d}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
