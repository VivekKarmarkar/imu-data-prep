#!/usr/bin/env python3
"""
Create a cumulative dataset combining all three sources (Movella, Sassari, Seel)
with randomized trial indices. Individual dataset outputs are not modified.
"""

import shutil
import numpy as np
import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
CUMULATIVE_DIR = BASE_DIR / "cumulative_output"

SOURCES = [
    ("movella_output", 0, 63),
    ("sassari_output", 64, 194),
    ("seel_output", 195, 271),
]

SEED = 42


def main():
    input_dir = CUMULATIVE_DIR / "input_data"
    input_plot_dir = input_dir / "plots"
    ref_dir = CUMULATIVE_DIR / "reference_trials"
    ref_plot_dir = ref_dir / "plots"
    method_dir = CUMULATIVE_DIR / "data_prep_method"
    for d in [input_dir, input_plot_dir, ref_dir, ref_plot_dir, method_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect all original indices
    all_original = []
    for source_name, start, end in SOURCES:
        for idx in range(start, end + 1):
            all_original.append((source_name, idx))

    total = len(all_original)
    print(f"Total trials: {total}")

    # Create randomized mapping: new_index -> (source, old_index)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(total)

    mapping = []
    for new_idx, perm_idx in enumerate(perm):
        source_name, old_idx = all_original[perm_idx]
        mapping.append({
            "new_index": new_idx,
            "old_index": old_idx,
            "source": source_name,
        })

    # Save the mapping for traceability
    mapping_df = pd.DataFrame(mapping)
    mapping_df.to_csv(CUMULATIVE_DIR / "index_mapping.csv", index=False)
    print(f"Saved index mapping to index_mapping.csv")

    # Copy and rename files
    for entry in mapping:
        new_idx = entry["new_index"]
        old_idx = entry["old_index"]
        source = entry["source"]
        src_dir = BASE_DIR / source

        # Input data CSV
        src_data = src_dir / "input_data" / f"trial_{old_idx:03d}.csv"
        dst_data = input_dir / f"trial_{new_idx:03d}.csv"
        shutil.copy2(str(src_data), str(dst_data))

        # Input meta CSV — update trial_index
        src_meta = src_dir / "input_data" / f"trial_{old_idx:03d}_meta.csv"
        dst_meta = input_dir / f"trial_{new_idx:03d}_meta.csv"
        meta = pd.read_csv(src_meta)
        meta["trial_index"] = new_idx
        meta.to_csv(dst_meta, index=False)

        # Input plot
        src_plot = src_dir / "input_data" / "plots" / f"trial_{old_idx:03d}_gyro.png"
        dst_plot = input_plot_dir / f"trial_{new_idx:03d}_gyro.png"
        if src_plot.exists():
            shutil.copy2(str(src_plot), str(dst_plot))

        # Reference CSV
        src_ref = src_dir / "reference_trials" / f"trial_{old_idx:03d}_ref.csv"
        dst_ref = ref_dir / f"trial_{new_idx:03d}_ref.csv"
        shutil.copy2(str(src_ref), str(dst_ref))

        # Reference plot
        src_ref_plot = src_dir / "reference_trials" / "plots" / f"trial_{old_idx:03d}_ref.png"
        dst_ref_plot = ref_plot_dir / f"trial_{new_idx:03d}_ref.png"
        if src_ref_plot.exists():
            shutil.copy2(str(src_ref_plot), str(dst_ref_plot))

    # Verify
    data_csvs = sorted(input_dir.glob("trial_*.csv"))
    data_only = [f for f in data_csvs if "_meta" not in f.name]
    ref_csvs = sorted(ref_dir.glob("trial_*_ref.csv"))
    print(f"\nCumulative dataset created:")
    print(f"  input_data: {len(data_only)} trials ({data_only[0].name} – {data_only[-1].name})")
    print(f"  reference_trials: {len(ref_csvs)} files")

    # Show a few mapping examples
    print(f"\nSample mapping (new -> source:old):")
    for entry in mapping[:10]:
        print(f"  {entry['new_index']:03d} -> {entry['source']}:{entry['old_index']:03d}")
    print(f"  ...")

    # Source distribution check
    source_counts = mapping_df["source"].value_counts()
    print(f"\nSource distribution:")
    for src, count in source_counts.items():
        print(f"  {src}: {count} trials")


if __name__ == "__main__":
    main()
