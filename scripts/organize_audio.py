import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_algorithm_slug(algorithm: str, intervention_value: float = None, intervention_frequency: int = None) -> str:
    """Create a filesystem-safe slug for algorithm with parameters."""

    # Clean up algorithm name
    if algorithm.startswith("intervention_"):
        base_algo = algorithm.replace("intervention_", "")
        # Add parameters for intervention algorithms
        if intervention_value is not None and intervention_frequency is not None:
            slug = f"{base_algo}_val{intervention_value}_freq{intervention_frequency}"
        else:
            slug = base_algo
    else:
        # Non-intervention algorithms (musicgen, sae_passthrough)
        slug = algorithm

    # Make filesystem safe
    slug = re.sub(r"[^\w\-_\.]", "_", slug)
    slug = re.sub(r"_+", "_", slug)  # Replace multiple underscores with single
    slug = slug.strip("_")

    return slug


def load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """Load metadata from a run directory."""
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        logger.warning(f"No metadata.json found in {run_dir}")
        return {}

    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata from {run_dir}: {e}")
        return {}


def find_run_directories(base_dir: Path) -> List[Path]:
    """Find all run directories containing metadata.json files."""
    run_dirs = []

    for item in base_dir.iterdir():
        if item.is_dir() and (item / "metadata.json").exists():
            run_dirs.append(item)

    logger.info(f"Found {len(run_dirs)} run directories")
    return run_dirs


def collect_audio_files(run_dirs: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Collect all audio files grouped by algorithm slug."""

    algorithm_files = defaultdict(list)

    for run_dir in run_dirs:
        logger.info(f"Processing run: {run_dir.name}")

        # Load metadata to get run parameters
        metadata = load_run_metadata(run_dir)
        execution_params = metadata.get("execution_params", {})

        run_intervention_value = execution_params.get("intervention_value", 0.0)
        run_intervention_frequency = execution_params.get("intervention_frequency", 1)

        # Walk through all audio files in the run
        for audio_file in run_dir.rglob("*.wav"):
            # Parse the path to extract algorithm info
            # Expected structure: run_dir/feature/algorithm/audio_file.wav
            path_parts = audio_file.relative_to(run_dir).parts

            if len(path_parts) >= 3:
                feature = path_parts[0]  # e.g., "f1393"
                algorithm = path_parts[1]  # e.g., "intervention_set_value"
                filename = path_parts[2]  # e.g., "p00_s00.wav"

                # Create algorithm slug with parameters
                algo_slug = create_algorithm_slug(algorithm, run_intervention_value, run_intervention_frequency)

                # Store file info
                file_info = {
                    "source_path": audio_file,
                    "feature": feature,
                    "algorithm": algorithm,
                    "original_filename": filename,
                    "run_dir": run_dir.name,
                    "run_intervention_value": run_intervention_value,
                    "run_intervention_frequency": run_intervention_frequency,
                }

                algorithm_files[algo_slug].append(file_info)
            else:
                logger.warning(f"Unexpected path structure: {audio_file}")

    return algorithm_files


def copy_and_organize_files(algorithm_files: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Copy and organize audio files into algorithm directories."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each algorithm
    for algo_slug, files in algorithm_files.items():
        logger.info(f"Processing algorithm: {algo_slug} ({len(files)} files)")

        # Create algorithm directory
        algo_dir = output_dir / algo_slug
        algo_dir.mkdir(exist_ok=True)

        # Copy files with sequential naming
        for i, file_info in enumerate(files):
            source_path = file_info["source_path"]

            # Create new filename: simple sequential numbering
            new_filename = f"{i:04d}.wav"
            dest_path = algo_dir / new_filename

            # Copy the file
            try:
                shutil.copy2(source_path, dest_path)
                logger.debug(f"Copied {source_path} -> {dest_path}")
            except Exception as e:
                logger.error(f"Error copying {source_path}: {e}")
                continue

        # Create a mapping file for reference
        mapping_file = algo_dir / "file_mapping.json"
        mapping_data = []

        for i, file_info in enumerate(files):
            mapping_entry = {
                "new_filename": f"{i:04d}.wav",
                "original_path": str(file_info["source_path"]),
                "feature": file_info["feature"],
                "algorithm": file_info["algorithm"],
                "original_filename": file_info["original_filename"],
                "run_dir": file_info["run_dir"],
                "run_intervention_value": file_info["run_intervention_value"],
                "run_intervention_frequency": file_info["run_intervention_frequency"],
            }
            mapping_data.append(mapping_entry)

        with open(mapping_file, "w") as f:
            json.dump(mapping_data, f, indent=2)

        logger.info(f"Created {algo_dir} with {len(files)} files and mapping")


def create_summary_report(algorithm_files: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Create a summary report of the organization."""

    summary = {
        "total_algorithms": len(algorithm_files),
        "total_files": sum(len(files) for files in algorithm_files.values()),
        "algorithms": {},
    }

    for algo_slug, files in algorithm_files.items():
        # Group by features and runs for statistics
        features = set(f["feature"] for f in files)
        runs = set(f["run_dir"] for f in files)

        # Get algorithm info from first file
        first_file = files[0]

        summary["algorithms"][algo_slug] = {
            "file_count": len(files),
            "features": sorted(list(features)),
            "feature_count": len(features),
            "runs": sorted(list(runs)),
            "run_count": len(runs),
            "base_algorithm": first_file["algorithm"],
            "intervention_value": first_file["run_intervention_value"],
            "intervention_frequency": first_file["run_intervention_frequency"],
        }

    # Save summary
    summary_file = output_dir / "organization_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary to console
    print(f"\n{'=' * 60}")
    print("AUDIO ORGANIZATION SUMMARY")
    print("=" * 60)
    print(f"Total algorithms: {summary['total_algorithms']}")
    print(f"Total files organized: {summary['total_files']}")
    print(f"Output directory: {output_dir}")

    print("\nAlgorithms created:")
    for algo_slug, info in summary["algorithms"].items():
        print(
            f"  {algo_slug:30} | {info['file_count']:3d} files | {info['feature_count']:2d} features | {info['run_count']:2d} runs"
        )


def main():
    parser = argparse.ArgumentParser(description="Organize audio files by algorithm from intervention runs")
    parser.add_argument("input_dir", type=Path, help="Directory containing intervention run directories")
    parser.add_argument(
        "--output",
        type=Path,
        default="organized_audio",
        help="Output directory for organized files (default: organized_audio)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually copying files")

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    # Find all run directories
    run_dirs = find_run_directories(args.input_dir)

    if not run_dirs:
        logger.error("No run directories found with metadata.json files")
        return

    # Collect all audio files grouped by algorithm
    algorithm_files = collect_audio_files(run_dirs)

    if not algorithm_files:
        logger.error("No audio files found in run directories")
        return

    logger.info(f"Found {len(algorithm_files)} unique algorithms")

    if args.dry_run:
        # Just show what would be done
        print(f"\n{'=' * 60}")
        print("DRY RUN - Would create the following structure:")
        print("=" * 60)

        for algo_slug, files in algorithm_files.items():
            features = set(f["feature"] for f in files)
            print(f"{algo_slug}/ ({len(files)} files from {len(features)} features)")
            for i in range(min(5, len(files))):  # Show first 5 files as example
                print(f"  {i:04d}.wav (from {files[i]['source_path']})")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
    else:
        # Actually copy and organize files
        copy_and_organize_files(algorithm_files, args.output)

        # Create summary report
        create_summary_report(algorithm_files, args.output)

        logger.info("Audio organization complete!")


if __name__ == "__main__":
    main()
