# IMU Data Prep

Benchmark dataset preparation pipeline for inertial orientation estimation. Processes three heterogeneous IMU source datasets into a unified, trial-based format: 272 self-contained movement trials with gyroscope/accelerometer sensor data, reference quaternion ground truth, and boundary conditions.

## Overview

This project takes raw IMU recordings from three independent research datasets — each with different sensors, sampling rates, file formats, and labeling conventions — and produces a single standardized benchmark. Each trial is a time-windowed movement segment with static padding, paired with optical motion capture reference quaternions for evaluation.

The prepared dataset is designed for testing quaternionic differential equation solvers in a separate project. This repo handles only data preparation and verification.

## Source Datasets

| Dataset | Sensors | Sample Rate | Format | Trials |
|---------|---------|-------------|--------|--------|
| **Movella** | 4 Xsens IMUs on rigid body | 400 Hz | `.txt` | 64 |
| **Sassari** | 6 mixed IMUs × 3 speeds | 100 Hz | `.mat` (MATLAB v5) | 131 |
| **Seel (BROAD)** | 39 recordings, single sensor | 285.7 Hz | `.mat` + `.hdf5` | 77 |

**Total: 272 trials**

## Output Format

Every trial, regardless of source, follows this uniform structure:

```
trial_XXX.csv          # time, Gyr_X, Gyr_Y, Gyr_Z, Acc_X, Acc_Y, Acc_Z
trial_XXX_meta.csv     # boundary quaternions (q_init, q_final), timestamps, movement window
trial_XXX_ref.csv      # time, qw, qx, qy, qz (full reference quaternion time series)
```

All trials are time-normalized to start at t=0 and include ~0.5s of static padding on each side of the movement.

## Pipeline

### Per-Dataset Processing

Each source dataset has its own processing script that handles format-specific parsing, movement detection, trial extraction, and output generation:

```bash
python prepare_movella.py    # → movella_output/   (trials 000–063)
python prepare_sassari.py    # → sassari_output/    (trials 064–194)
python prepare_seel.py       # → seel_output/       (trials 195–271)
```

Movella and Sassari detect movements via smoothed gyroscope magnitude thresholding (0.05 rad/s). Seel uses pre-labeled movement masks from the BROAD dataset.

### Movella Post-Processing

The Movella pipeline uses additional scripts (run in order after `prepare_movella.py`):

```bash
python extract_reference_trials.py   # extract reference quaternion time series
python renumber_trials.py            # global indices across 4 sensors
python merge_sensor_folders.py       # consolidate per-sensor folders
python normalize_times.py            # shift all times to start at t=0
```

Sassari and Seel scripts handle these steps internally.

### Cumulative Dataset

Combines all 272 trials into a single randomized dataset with no telltale of source:

```bash
python create_cumulative.py          # copy + randomize indices (seed=42)
python regenerate_cumulative_plots.py # fresh plots with new indices
```

The cumulative dataset includes `index_mapping.csv` for traceability (maps randomized index back to source dataset and original trial number).

## Project Structure

```
├── movella_dataset/          # Raw Xsens .txt files + reference
├── sassari_dataset/          # Raw .mat files + videos + paper
├── seel_dataset/             # BROAD dataset (.mat, .hdf5, animations)
├── prepare_movella.py        # Movella processing pipeline
├── prepare_sassari.py        # Sassari processing pipeline
├── prepare_seel.py           # Seel/BROAD processing pipeline
├── create_cumulative.py      # Cumulative dataset builder
├── regenerate_cumulative_plots.py
├── *_output/                 # Generated outputs (gitignored)
│   ├── input_data/           # Trial CSVs + meta + plots
│   ├── reference_trials/     # Reference quaternions + plots
│   └── data_prep_method/     # Methodology docs + script copies
└── cumulative_output/        # Randomized combined dataset (gitignored)
    └── index_mapping.csv     # Traceability: new → (source, old)
```

## Getting Started

### Prerequisites

- Python >= 3.7
- NVIDIA GPU recommended (not required)

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scipy h5py
```

### Generate All Outputs

```bash
# Process each dataset (order doesn't matter)
python prepare_movella.py
python extract_reference_trials.py
python renumber_trials.py
python merge_sensor_folders.py
python normalize_times.py

python prepare_sassari.py
python prepare_seel.py

# Build cumulative dataset
python create_cumulative.py
python regenerate_cumulative_plots.py
```

Output directories (`*_output/`) are gitignored and fully regenerable from the source data and scripts.

## Verification

Each processing script generates verification plots:
- **Gyroscope plots**: Gyr_X/Y/Z with movement start/end markers and boundary quaternion timestamps
- **Reference plots**: Quaternion components (qw, qx, qy, qz) with norm verification (should be ≡ 1.0)

Plots are stored in `<dataset>_output/input_data/plots/` and `<dataset>_output/reference_trials/plots/`. Each dataset's `data_prep_method/` folder contains a methodology document and copies of the processing scripts used.

## Seel/BROAD Dataset

The `seel_dataset/` directory contains the [BROAD benchmark](https://doi.org/10.3390/data5030063) by Laidig et al., available under CC-BY-4.0. See `seel_dataset/README.md` and `seel_dataset/LICENSE` for details.
