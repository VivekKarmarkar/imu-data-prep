#!/usr/bin/env python3
"""
Shift all Sassari trial indices by 64 so they continue from where Movella ends.
Movella: 0–63, Sassari: 64–194
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "sassari_output"
INPUT_DIR = OUTPUT_DIR / "input_data"
REF_DIR = OUTPUT_DIR / "reference_trials"
OFFSET = 64
OLD_COUNT = 131  # 0–130


def main():
    # Rename in reverse order to avoid collisions
    for old_idx in range(OLD_COUNT - 1, -1, -1):
        new_idx = old_idx + OFFSET

        # Input data CSV
        old_data = INPUT_DIR / f"trial_{old_idx:03d}.csv"
        new_data = INPUT_DIR / f"trial_{new_idx:03d}.csv"
        if old_data.exists():
            old_data.rename(new_data)

        # Input meta CSV — update trial_index inside
        old_meta = INPUT_DIR / f"trial_{old_idx:03d}_meta.csv"
        new_meta = INPUT_DIR / f"trial_{new_idx:03d}_meta.csv"
        if old_meta.exists():
            meta = pd.read_csv(old_meta)
            meta["trial_index"] = new_idx
            meta.to_csv(new_meta, index=False)
            if old_meta != new_meta:
                old_meta.unlink()

        # Input plot
        old_plot = INPUT_DIR / "plots" / f"trial_{old_idx:03d}_gyro.png"
        new_plot = INPUT_DIR / "plots" / f"trial_{new_idx:03d}_gyro.png"
        if old_plot.exists():
            old_plot.rename(new_plot)

        # Reference CSV
        old_ref = REF_DIR / f"trial_{old_idx:03d}_ref.csv"
        new_ref = REF_DIR / f"trial_{new_idx:03d}_ref.csv"
        if old_ref.exists():
            old_ref.rename(new_ref)

        # Reference plot
        old_ref_plot = REF_DIR / "plots" / f"trial_{old_idx:03d}_ref.png"
        new_ref_plot = REF_DIR / "plots" / f"trial_{new_idx:03d}_ref.png"
        if old_ref_plot.exists():
            old_ref_plot.rename(new_ref_plot)

    print(f"Shifted {OLD_COUNT} trials by offset {OFFSET}")
    print(f"New range: {OFFSET:03d} – {OFFSET + OLD_COUNT - 1:03d}")

    # Verify
    data_files = sorted(INPUT_DIR.glob("trial_*.csv"))
    data_only = [f for f in data_files if "_meta" not in f.name]
    print(f"Input data files: {len(data_only)} ({data_only[0].name} – {data_only[-1].name})")

    ref_files = sorted(REF_DIR.glob("trial_*_ref.csv"))
    print(f"Reference files: {len(ref_files)} ({ref_files[0].name} – {ref_files[-1].name})")


if __name__ == "__main__":
    main()
