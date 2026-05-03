# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Benchmark dataset preparation for IMU (Inertial Measurement Unit) orientation estimation. Three heterogeneous source datasets are processed into a uniform trial-based format: sensor data (gyroscope + accelerometer) paired with reference quaternion ground truth and boundary conditions. The prepared data is used in a **separate project** for testing quaternionic differential equation methods — this repo handles data prep and evaluation only.

## Environment

```bash
source .venv/bin/activate
# Dependencies: numpy, pandas, matplotlib, scipy, h5py
```

## Architecture

The pipeline has three independent dataset processors, then post-processing scripts that unify them:

### Dataset Processors (run independently, in order)
- `prepare_movella.py` → `movella_output/` — 4 Xsens sensors, 400 Hz, .txt format, gyro-based movement detection
- `prepare_sassari.py` → `sassari_output/` — 6 mixed sensors × 3 speeds, 100 Hz, .mat format, gyro-based detection
- `prepare_seel.py` → `seel_output/` — 39 recordings, 285.7 Hz, .mat format, pre-labeled movement mask

### Post-processing (Movella-specific, run after prepare_movella.py)
- `extract_reference_trials.py` — extracts reference quaternion time series per trial
- `renumber_trials.py` — assigns global indices across 4 sensors (0–63)
- `merge_sensor_folders.py` — consolidates per-sensor folders into single `input_data/`
- `normalize_times.py` — shifts all trial times to start at t=0

Sassari and Seel scripts handle renumbering, merging, and normalization internally.

### Cumulative Dataset
- `create_cumulative.py` — copies all 272 trials into `cumulative_output/` with randomized indices (seed=42)
- `regenerate_cumulative_plots.py` — regenerates all plots with new indices (no source dataset telltale)
- `shift_sassari_indices.py` — offsets Sassari trials by 64 to continue from Movella

### Global Index Ranges
| Dataset | Range | Count |
|---------|-------|-------|
| Movella | 000–063 | 64 |
| Sassari | 064–194 | 131 |
| Seel | 195–271 | 77 |

## Output Structure (uniform across all datasets)

```
<dataset>_output/
  input_data/
    trial_XXX.csv          # time, Gyr_X/Y/Z, Acc_X/Y/Z
    trial_XXX_meta.csv     # boundary quaternions, timestamps
    plots/
  reference_trials/
    trial_XXX_ref.csv      # time, qw, qx, qy, qz
    plots/
  data_prep_method/        # methodology doc + script copies
```

## Key Design Decisions

- Each trial is self-contained: padded sensor segment + boundary quaternions from reference
- ~0.5s static padding on each side of every movement segment
- Cumulative dataset has `index_mapping.csv` for traceability (new_index → source + old_index)
- Individual dataset outputs are **never modified** by cumulative or cross-dataset scripts
- Movement detection uses smoothed gyroscope magnitude threshold (0.05 rad/s) for Movella/Sassari; Seel uses pre-labeled masks
