#!/usr/bin/env python3
"""
Intervention Evaluation Script

This script evaluates SAE intervention results by:
1. Calculating CLAP scores between prompts and generated audio
2. Saving individual file scores
3. Creating aggregations by algorithm and feature

Usage:
    python evaluate_interventions.py /path/to/generation/run --output eval_results.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import statistics

import torch
import torchaudio
from transformers import ClapModel, ClapProcessor
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_audio_file(file_path: Path, target_sr: int = 48000) -> torch.Tensor:
    """Load and resample audio file for CLAP (48kHz)."""
    try:
        audio_tensor, sr = torchaudio.load(str(file_path))
        if sr != target_sr:
            transform = torchaudio.transforms.Resample(sr, target_sr)
            audio_tensor = transform(audio_tensor)
        return audio_tensor[0]  # Take first channel
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


@torch.no_grad()
def embed_text(processor: ClapProcessor, model: ClapModel, text: str, device: torch.device) -> torch.Tensor:
    """Embed text using CLAP model."""
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu()


@torch.no_grad()
def embed_audio(
    processor: ClapProcessor,
    model: ClapModel,
    audio_tensor: torch.Tensor,
    device: torch.device,
    sampling_rate: int = 48000,
) -> torch.Tensor:
    """Embed single audio tensor using CLAP model."""
    audio_np = audio_tensor.numpy()
    inputs = processor(audios=[audio_np], sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    a_emb = model.get_audio_features(**inputs)
    a_emb = a_emb / a_emb.norm(dim=-1, keepdim=True)
    return a_emb.squeeze(0).cpu()


def calculate_clap_score(
    prompt: str, audio_path: Path, processor: ClapProcessor, model: ClapModel, device: torch.device
) -> float:
    """Calculate CLAP similarity score between prompt and audio file."""
    # Embed text prompt
    text_emb = embed_text(processor, model, prompt, device)

    # Load and embed audio
    audio_tensor = load_audio_file(audio_path, target_sr=48000)  # CLAP uses 48kHz
    audio_emb = embed_audio(processor, model, audio_tensor, device)

    # Calculate similarity
    similarity = (audio_emb @ text_emb).item()
    return similarity


def parse_algorithm_from_path(filepath: str) -> str:
    """Extract algorithm name from filepath."""
    parts = filepath.split("/")
    for part in parts:
        if part.startswith("intervention_"):
            return part
        elif part in ["musicgen", "sae_passthrough"]:
            return part
    return "unknown"


def parse_feature_from_path(filepath: str) -> str:
    """Extract feature name from filepath."""
    parts = filepath.split("/")
    for part in parts:
        if part.startswith("f") and len(part) == 5:  # f0437 format
            return part
    return "unknown"


def load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """Load metadata from run directory."""
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        return json.load(f)


def evaluate_run(run_dir: Path, processor: ClapProcessor, model: ClapModel, device: torch.device) -> Dict[str, Any]:
    """Evaluate all files in a run directory."""
    logger.info(f"Evaluating run: {run_dir.name}")

    # Load metadata
    metadata = load_run_metadata(run_dir)

    # Calculate CLAP scores for each file
    individual_scores = []

    for song_entry in tqdm(metadata["songs"], desc="Calculating CLAP scores"):
        filepath = song_entry["filepath"]
        prompt = song_entry["prompt"]

        # Construct full path
        audio_path = run_dir / filepath

        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            continue

        try:
            # Calculate CLAP score
            clap_score = calculate_clap_score(prompt, audio_path, processor, model, device)

            # Extract algorithm and feature
            algorithm = parse_algorithm_from_path(filepath)
            feature = parse_feature_from_path(filepath)

            score_entry = {
                "filepath": filepath,
                "prompt": prompt,
                "clap_score": clap_score,
                "algorithm": algorithm,
                "feature": feature,
            }

            individual_scores.append(score_entry)

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            continue

    return {"metadata": metadata, "individual_scores": individual_scores}


def create_aggregations(individual_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create aggregations by algorithm and feature."""

    # Group by algorithm
    by_algorithm = defaultdict(list)
    for score in individual_scores:
        by_algorithm[score["algorithm"]].append(score["clap_score"])

    # Group by feature (only intervention algorithms)
    by_feature = defaultdict(list)
    for score in individual_scores:
        if score["algorithm"].startswith("intervention_"):
            by_feature[score["feature"]].append(score["clap_score"])

    # Group by feature-algorithm combination
    by_feature_algorithm = defaultdict(list)
    for score in individual_scores:
        key = f"{score['feature']}_{score['algorithm']}"
        by_feature_algorithm[key].append(score["clap_score"])

    # Calculate statistics
    def calc_stats(scores):
        if not scores:
            return None
        return {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }

    # Algorithm aggregations
    algorithm_stats = {}
    for algorithm, scores in by_algorithm.items():
        algorithm_stats[algorithm] = calc_stats(scores)

    # Feature aggregations (intervention only)
    feature_stats = {}
    for feature, scores in by_feature.items():
        feature_stats[feature] = calc_stats(scores)

    # Feature-algorithm aggregations
    feature_algorithm_stats = {}
    for feature_algo, scores in by_feature_algorithm.items():
        feature_algorithm_stats[feature_algo] = calc_stats(scores)

    return {
        "by_algorithm": algorithm_stats,
        "by_feature": feature_stats,
        "by_feature_algorithm": feature_algorithm_stats,
        "overall_stats": calc_stats([s["clap_score"] for s in individual_scores]),
    }


def save_results(results: Dict[str, Any], output_path: Path):
    """Save evaluation results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def print_summary(aggregations: Dict[str, Any]):
    """Print a summary of the evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print("\nALGORITHM PERFORMANCE:")
    print("-" * 40)
    algo_stats = aggregations["by_algorithm"]
    for algorithm, stats in sorted(algo_stats.items()):
        if stats:
            print(f"{algorithm:25} | Mean: {stats['mean']:.4f} | Count: {stats['count']}")

    print("\nFEATURE PERFORMANCE (Interventions Only):")
    print("-" * 40)
    feature_stats = aggregations["by_feature"]
    for feature, stats in sorted(feature_stats.items()):
        if stats:
            print(f"{feature:25} | Mean: {stats['mean']:.4f} | Count: {stats['count']}")

    print("\nFEATURE-ALGORITHM COMBINATIONS:")
    print("-" * 60)
    feature_algo_stats = aggregations["by_feature_algorithm"]
    for feature_algo, stats in sorted(feature_algo_stats.items()):
        if stats:
            print(f"{feature_algo:40} | Mean: {stats['mean']:.4f} | Count: {stats['count']}")

    overall = aggregations["overall_stats"]
    if overall:
        print("\nOVERALL STATISTICS:")
        print("-" * 40)
        print(f"Mean CLAP Score: {overall['mean']:.4f}")
        print(f"Median: {overall['median']:.4f}")
        print(f"Std Dev: {overall['std']:.4f}")
        print(f"Range: [{overall['min']:.4f}, {overall['max']:.4f}]")
        print(f"Total Files: {overall['count']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAE intervention results using CLAP scores")
    parser.add_argument("run_dir", type=Path, help="Path to generation run directory")
    parser.add_argument(
        "--output",
        type=Path,
        default="evaluation_results.json",
        help="Output JSON file path (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--clap-model", default="laion/clap-htsat-fused", help="CLAP model to use (default: laion/clap-htsat-fused)"
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device for CLAP model (default: cuda:0)")
    parser.add_argument(
        "--individual-scores-file", type=Path, default=None, help="Save individual scores to separate JSON file"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.run_dir.exists() or not args.run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {args.run_dir}")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load CLAP model
    logger.info(f"Loading CLAP model: {args.clap_model}")
    processor = ClapProcessor.from_pretrained(args.clap_model)
    model = ClapModel.from_pretrained(args.clap_model).to(device)
    model.eval()

    # Evaluate the run
    evaluation_results = evaluate_run(args.run_dir, processor, model, device)

    # Create aggregations
    aggregations = create_aggregations(evaluation_results["individual_scores"])

    # Prepare final results
    final_results = {
        "run_info": {
            "run_dir": str(args.run_dir),
            "execution_params": evaluation_results["metadata"]["execution_params"],
        },
        "aggregations": aggregations,
        "evaluation_settings": {"clap_model": args.clap_model, "device": str(device)},
    }

    # Save individual scores if requested
    if args.individual_scores_file:
        individual_results = {
            "run_info": final_results["run_info"],
            "individual_scores": evaluation_results["individual_scores"],
        }
        save_results(individual_results, args.individual_scores_file)
        logger.info(f"Individual scores saved to: {args.individual_scores_file}")

    # Save main results
    save_results(final_results, args.output)

    # Print summary
    print_summary(aggregations)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
