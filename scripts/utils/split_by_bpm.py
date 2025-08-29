from pathlib import Path
import librosa
import argparse
import logging
from typing import Optional
from src.project_config import INPUT_DATA_DIR
import tqdm

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def estimate_bpm(filename: Path) -> float:
    y, sr = librosa.load(filename)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    if isinstance(tempo, list):
        return tempo[-1]

    return tempo


def split_songs_by_bpm(
    input_file: Path,
    slower_file: Path,
    faster_file: Path,
    threshold_bpm: float,
    only_histogram: bool = False,
    histogram_file: Optional[Path] = None,
) -> None:
    """
    Split a collection of songs into two groups based on tempo.

    Songs with BPM greater than the given threshold are placed in one group,
    and songs with BPM less than or equal to the threshold are placed in the other.
    """
    ...

    try:
        logger.info(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)

        original_count = len(df)
        logger.info(f"Loaded {original_count:,} rows from {input_file}")

        # Create mask for faster songs
        mask = pd.Series([False] * len(df))  # is faster than threshold
        skipped = 0

        bpms = []

        for idx, path in enumerate(tqdm.tqdm(df["path"])):
            p = INPUT_DATA_DIR / "mtg-jamendo" / "datashare-instruments" / path

            if not p.exists():
                skipped += 1

                if skipped % 1000 == 0:
                    print("Skipped", skipped, "of", idx)

                continue

            bpm = estimate_bpm(p)
            bpms.append(bpm)

            if bpm >= threshold_bpm:
                mask[idx] = True

        if only_histogram:
            with open(str(histogram_file), "w") as f:
                for item in bpms:
                    f.write(f"{item}\n")
            # return

        # Apply the filter
        faster_df = df[mask].reset_index(drop=True)
        faster_count = len(faster_df)
        slower_df = df[~mask].reset_index(drop=True)
        slower_count = len(slower_df)

        # Save the filtered results
        slower_file.parent.mkdir(parents=True, exist_ok=True)
        slower_df.to_csv(slower_file, index=False)
        faster_file.parent.mkdir(parents=True, exist_ok=True)
        faster_df.to_csv(faster_file, index=False)

        logger.info(
            f"Results: {faster_count:,} of {original_count:,} songs are faster, {slower_count:,} of {original_count:,} songs are slower. Skipped: {skipped} due to lack of file."
        )
        logger.info(f"Saved filtered data to: {slower_file} and {faster_file}")

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        raise
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise


def main() -> None:
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Split mtg-jamendo songs by bpm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/input.csv data/slower.csv data/faster.csv --bpm <number>
  %(prog)s input.csv slower.csv faster.csv --bpm 128 --only-histogram --histogram-file hist.png
        """,
    )

    parser.add_argument("input_file", type=Path, help="Path to input CSV file")

    parser.add_argument("slower_file", type=Path, help="Path to output CSV file with slower songs")

    parser.add_argument("faster_file", type=Path, help="Path to output CSV file with faster songs")

    parser.add_argument("-b", "--bpm", type=float, required=True, help="Threshold BPM")

    parser.add_argument("--histogram-file", type=Path, required=False, help="Path to output png file for histogram")

    parser.add_argument(
        "--only-histogram",
        action="store_true",
        help="Generate only histogram of songs BPMs",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input_file.exists():
        parser.error(f"Input file does not exist: {args.input_file}")

    if args.only_histogram:
        if args.histogram_file is None:
            parser.error("Histogram file should be provided, when --only-histogram flag is active")

    # Run the filtering
    split_songs_by_bpm(
        input_file=args.input_file,
        slower_file=args.slower_file,
        faster_file=args.faster_file,
        threshold_bpm=args.bpm,
        only_histogram=args.only_histogram,
        histogram_file=args.histogram_file,
    )


if __name__ == "__main__":
    main()
