#!/usr/bin/env python3
"""
Prepare movement trials from the Seel (BROAD) IMU dataset.

39 recording files, each with pre-labeled movement segments via the 'movement' array.
Single sensor per file, optical tracking reference quaternions.
Sampling rate: 285.7 Hz.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from scipy.ndimage import uniform_filter1d

DATA_DIR = Path(__file__).parent / "seel_dataset" / "data_mat"
OUTPUT_DIR = Path(__file__).parent / "seel_output"
GLOBAL_OFFSET = 195  # Movella: 0–63, Sassari: 64–194

PADDING_SAMPLES = 143  # ~0.5s at 285.7 Hz
MIN_MOVE_DURATION = 50


def load_seel_mat(path: Path):
    data = sio.loadmat(str(path))
    sr = float(data["sampling_rate"][0][0])
    n = data["imu_gyr"].shape[0]
    time = np.arange(n) / sr

    imu_df = pd.DataFrame({
        "time": time,
        "Gyr_X": data["imu_gyr"][:, 0],
        "Gyr_Y": data["imu_gyr"][:, 1],
        "Gyr_Z": data["imu_gyr"][:, 2],
        "Acc_X": data["imu_acc"][:, 0],
        "Acc_Y": data["imu_acc"][:, 1],
        "Acc_Z": data["imu_acc"][:, 2],
    })

    ref_df = pd.DataFrame({
        "time": time,
        "qw": data["opt_quat"][:, 0],
        "qx": data["opt_quat"][:, 1],
        "qy": data["opt_quat"][:, 2],
        "qz": data["opt_quat"][:, 3],
    })

    movement = data["movement"].flatten()
    return imu_df, ref_df, movement, sr


def extract_segments(movement, n_samples):
    diff = np.diff(movement.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if movement[0] == 1:
        starts = np.concatenate([[0], starts])
    if movement[-1] == 1:
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


def save_trial(seg, ref_seg, q_init, q_final, t_init, t_final,
               move_start_t, move_end_t, trial_idx, input_dir, ref_dir):
    seg.to_csv(input_dir / f"trial_{trial_idx:03d}.csv", index=False)
    ref_seg.to_csv(ref_dir / f"trial_{trial_idx:03d}_ref.csv", index=False)

    meta = pd.DataFrame([{
        "trial_index": trial_idx,
        "q_init_w": q_init[0], "q_init_x": q_init[1],
        "q_init_y": q_init[2], "q_init_z": q_init[3],
        "q_final_w": q_final[0], "q_final_x": q_final[1],
        "q_final_y": q_final[2], "q_final_z": q_final[3],
        "t_init": t_init, "t_final": t_final,
        "move_start_time": move_start_t, "move_end_time": move_end_t,
    }])
    meta.to_csv(input_dir / f"trial_{trial_idx:03d}_meta.csv", index=False)


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


def plot_overview(imu_df, movement, file_label, trials, plot_dir):
    mag = np.sqrt(imu_df["Gyr_X"]**2 + imu_df["Gyr_Y"]**2 + imu_df["Gyr_Z"]**2)
    smoothed = uniform_filter1d(mag.values, size=71)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(imu_df["time"].values, smoothed, color="steelblue", linewidth=0.5)
    for tr in trials:
        t_s = imu_df["time"].iloc[tr["padded_start"]]
        t_e = imu_df["time"].iloc[min(tr["padded_end"], len(imu_df)-1)]
        ax.axvspan(t_s, t_e, alpha=0.15, color="orange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Smoothed gyro magnitude (rad/s)")
    ax.set_title(f"{file_label} — Movement detection overview")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / f"overview_{file_label}.png", dpi=150)
    plt.close(fig)


def main():
    input_dir = OUTPUT_DIR / "input_data"
    input_plot_dir = input_dir / "plots"
    ref_dir = OUTPUT_DIR / "reference_trials"
    ref_plot_dir = ref_dir / "plots"
    method_dir = OUTPUT_DIR / "data_prep_method"
    for d in [input_dir, input_plot_dir, ref_dir, ref_plot_dir, method_dir]:
        d.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(DATA_DIR.glob("*.mat"))
    print(f"Found {len(mat_files)} recording files")

    trial_idx = GLOBAL_OFFSET

    for mat_path in mat_files:
        file_label = mat_path.stem
        print(f"\n{'='*60}")
        print(f"Processing: {file_label}")

        imu_df, ref_df, movement, sr = load_seel_mat(mat_path)
        n_samples = len(imu_df)
        print(f"  Samples: {n_samples}, duration: {n_samples/sr:.1f}s, rate: {sr:.1f} Hz")

        trials = extract_segments(movement, n_samples)
        print(f"  Movement segments: {len(trials)}")

        plot_overview(imu_df, movement, file_label, trials, input_plot_dir)

        for i, trial in enumerate(trials):
            seg = imu_df.iloc[trial["padded_start"]:trial["padded_end"]].copy()
            seg = seg[["time", "Gyr_X", "Gyr_Y", "Gyr_Z", "Acc_X", "Acc_Y", "Acc_Z"]].reset_index(drop=True)

            ref_seg = ref_df.iloc[trial["padded_start"]:trial["padded_end"]].copy()
            ref_seg = ref_seg[["time", "qw", "qx", "qy", "qz"]].reset_index(drop=True)

            move_start_t = imu_df["time"].iloc[trial["move_start"]]
            move_end_t = imu_df["time"].iloc[min(trial["move_end"], n_samples-1)]

            t_init = seg["time"].iloc[0]
            t_final = seg["time"].iloc[-1]

            q_init = ref_df.iloc[trial["padded_start"]][["qw", "qx", "qy", "qz"]].values.astype(float)
            q_final_idx = min(trial["padded_end"] - 1, len(ref_df) - 1)
            q_final = ref_df.iloc[q_final_idx][["qw", "qx", "qy", "qz"]].values.astype(float)

            # Normalize to t=0
            t0 = seg["time"].iloc[0]
            seg["time"] -= t0
            ref_seg["time"] -= t0
            move_start_t -= t0
            move_end_t -= t0
            t_init -= t0
            t_final -= t0

            save_trial(seg, ref_seg, q_init, q_final, t_init, t_final,
                       move_start_t, move_end_t, trial_idx, input_dir, ref_dir)
            plot_trial_gyro(seg, move_start_t, move_end_t, t_init, t_final,
                            trial_idx, input_plot_dir)
            plot_trial_ref(ref_seg, move_start_t, move_end_t, trial_idx, ref_plot_dir)

            print(f"  Trial {trial_idx}: t=[0, {seg['time'].iloc[-1]:.3f}]s, "
                  f"move=[{move_start_t:.3f}, {move_end_t:.3f}]s")
            trial_idx += 1

    total = trial_idx - GLOBAL_OFFSET
    print(f"\n{'='*60}")
    print(f"Done. Total trials: {total}")
    print(f"Trial range: {GLOBAL_OFFSET:03d} – {trial_idx-1:03d}")
    print(f"Global range across all datasets: 000 – {trial_idx-1:03d}")


if __name__ == "__main__":
    main()
