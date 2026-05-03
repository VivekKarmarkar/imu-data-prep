#!/usr/bin/env python3
"""
Merge all 4 sensor trial folders into a single 'input_data' folder
with a 'plots' subfolder, then remove the individual sensor folders.
"""

import shutil
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "movella_output"
INPUT_DATA_DIR = OUTPUT_DIR / "input_data"


def main():
    INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (INPUT_DATA_DIR / "plots").mkdir(parents=True, exist_ok=True)

    sensor_dirs = sorted(OUTPUT_DIR.glob("MT_*"))
    print(f"Found {len(sensor_dirs)} sensor directories to merge")

    for sensor_dir in sensor_dirs:
        print(f"\nMerging {sensor_dir.name}:")

        for csv_file in sorted(sensor_dir.glob("trial_*.csv")):
            dest = INPUT_DATA_DIR / csv_file.name
            shutil.move(str(csv_file), str(dest))
            print(f"  {csv_file.name}")

        plot_dir = sensor_dir / "plots"
        if plot_dir.exists():
            for plot_file in sorted(plot_dir.glob("*.png")):
                dest = INPUT_DATA_DIR / "plots" / plot_file.name
                shutil.move(str(plot_file), str(dest))

        # Remove the now-empty sensor directory
        shutil.rmtree(sensor_dir)
        print(f"  Removed {sensor_dir.name}/")

    # Verify
    csvs = sorted(INPUT_DATA_DIR.glob("trial_*.csv"))
    plots = sorted((INPUT_DATA_DIR / "plots").glob("*.png"))
    print(f"\n{'='*60}")
    print(f"input_data/: {len(csvs)} CSV files, {len(plots)} plots")
    print(f"Trial range: {csvs[0].name} – {csvs[-1].name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
