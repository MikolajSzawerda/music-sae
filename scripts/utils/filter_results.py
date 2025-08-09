import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Filter results by filter")
    parser.add_argument("input_file", type=Path, help="Input JSON file with individual scores")
    parser.add_argument("scores_files", type=Path, help="Scores JSON file with individual scores")
    parser.add_argument(
        "--output",
        type=Path,
        default="expanded_descriptions.json",
        help="Output JSON file path (default: expanded_descriptions.json)",
    )
    parser.add_argument("--n", type=int, default=10)
    # parser.add_argument("--features", nargs='+')

    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    # filter_features = set([f'f{int(x):04d}' for x in args.features])
    with open(args.scores_files, "r") as f:
        filter_features = [x[0] for x in sorted(json.load(f).items(), key=lambda x: x[1], reverse=True)[: args.n]]
    with open(args.input_file, "r") as f:
        data = json.load(f)
    logger.info(f"Filtered features: {filter_features}")
    filtered_scores = [x for x in data["individual_scores"] if x["feature"] in filter_features]
    data["individual_scores"] = filtered_scores
    data["run_info"]["execution_params"]["features"] = [int(x[1:]) for x in filter_features]
    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
