import argparse
import csv
import io
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torchaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from transformers import ClapModel, ClapProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Concept(BaseModel):
    """Represents a single identified musical concept."""

    name: str = Field(..., description="Concise name for the musical concept.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1).")
    description: str = Field(..., description="Brief description of the concept.")


class ConceptLabels(BaseModel):
    """Represents the overall analysis result for a set of audio clips."""

    concepts: List[Concept] = Field(..., description="List of specific concepts identified.")
    overall_summary: str = Field(..., description="Concise description of the shared concept.")
    overall_name: str = Field(..., description="Concise name for the shared concept.")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score (0-1).")


GEMINI_PROMPT = """
Listen very carefully to this set of audio clips, which consists of song snippets concatenated in random order. You need to discover common musical patterns across the whole set, to identify what musical feature is shared across all clips. You will need to listen carefully. For each potential concept you identify, output a name, a confidence score between 0 and 1 (where 1 is highest confidence), and a concise description of the concept.
At a higher level, describe the overall concept shared across the set, give it a suitable name, and provide an overall confidence score (0 to 1).
Describe the **underlying concepts** not the specific audio snippets (e.g. your description could say "the concept" but not "the audio snippets"). However, try to avoid such verbiage altogether and concisely describe the musical concept's main attributes.
Include NO FILLER text.
Focus on being specific. Concepts could relate to genre (e.g., hip-hop, salsa, reggaeton, balkan), instruments (e.g., piano, cello, guitar, flute), recording/production techniques (e.g., reverberation, drones, noise, DJ scratching, beatboxing, drum machine, hi-hat patterns, fingerpicking, live recording artifacts, low-pass filtering), or more nuanced musical ideas (e.g., drum solo, chill dance rhythm, serene woodwinds arrangement). These are illustrative examples, NOT a fixed list to choose from.
""".strip()


def load_audio_file(file_path: Path, target_sr: int = 32000) -> torch.Tensor:
    """Load and resample audio file."""
    try:
        audio_tensor, sr = torchaudio.load(str(file_path))
        if sr != target_sr:
            transform = torchaudio.transforms.Resample(sr, target_sr)
            audio_tensor = transform(audio_tensor)
        return audio_tensor[0]  # Take first channel
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def get_audio_files(directory: Path) -> List[Path]:
    """Get all audio files from a directory."""
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    audio_files = []

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_files.append(file_path)

    return audio_files


def concatenate_and_convert_to_mp3(audio_files: List[Path], target_sr: int = 32000) -> bytes:
    """Concatenate audio files and convert to MP3 bytes."""
    audios = []
    for file_path in audio_files:
        audio_tensor = load_audio_file(file_path, target_sr)
        audios.append(audio_tensor)

    if not audios:
        raise ValueError("No audio files to concatenate")

    concatenated = torch.cat(audios, dim=0)

    # Convert to MP3 bytes
    buffer = io.BytesIO()
    torchaudio.save(buffer, concatenated.unsqueeze(0), target_sr, format="mp3")
    return buffer.getvalue()


def query_gemini(audio_bytes: bytes, model_name: str = "gemini-1.5-flash") -> ConceptLabels:
    """Query Gemini with audio and get structured response."""
    api_key = os.getenv("GENMINI_API")
    if not api_key:
        raise ValueError("GENMINI_API environment variable not set")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=[
            GEMINI_PROMPT,
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/mp3",
            ),
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": ConceptLabels,
        },
    )

    return ConceptLabels.model_validate(json.loads(response.text))


@torch.no_grad()
def embed_text(processor: ClapProcessor, model: ClapModel, text: str, device: torch.device) -> torch.Tensor:
    """Embed text using CLAP model."""
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu()


@torch.no_grad()
def embed_audios(
    processor: ClapProcessor,
    model: ClapModel,
    audio_tensors: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 10,
    sampling_rate: int = 48000,
) -> torch.Tensor:
    """Embed audio tensors using CLAP model."""
    embs = []
    for i in range(0, len(audio_tensors), batch_size):
        batch = [tensor.numpy() for tensor in audio_tensors[i : i + batch_size]]
        inputs = processor(audios=batch, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        a_emb = model.get_audio_features(**inputs)
        a_emb = a_emb / a_emb.norm(dim=-1, keepdim=True)
        embs.append(a_emb.cpu())
    return torch.cat(embs, dim=0)


def calculate_clap_score(
    description: str, audio_files: List[Path], processor: ClapProcessor, model: ClapModel, device: torch.device
) -> float:
    """Calculate CLAP similarity score between description and audio files."""
    # Embed text description
    text_emb = embed_text(processor, model, description, device)

    # Load and embed audio files
    audio_tensors = []
    for audio_file in audio_files:
        audio_tensor = load_audio_file(audio_file, target_sr=48000)  # CLAP uses 48kHz
        audio_tensors.append(audio_tensor)

    if not audio_tensors:
        return 0.0

    # Embed audios
    audio_embs = embed_audios(processor, model, audio_tensors, device)

    # Calculate similarity
    similarities = audio_embs @ text_emb
    return similarities.mean().item()


def split_files(files: List[Path], train_percent: float) -> Tuple[List[Path], List[Path]]:
    """Split files into training and evaluation sets."""
    random.shuffle(files)
    split_idx = int(len(files) * train_percent / 100)
    return files[:split_idx], files[split_idx:]


def process_feature_directory(
    feature_dir: Path,
    gemini_model: str,
    clap_processor: ClapProcessor,
    clap_model: ClapModel,
    device: torch.device,
    gemini_percent: float,
) -> Dict[str, str]:
    """Process a single feature directory."""
    logger.info(f"Processing feature directory: {feature_dir.name}")

    # Get all audio files
    audio_files = get_audio_files(feature_dir)
    if len(audio_files) < 2:
        logger.warning(f"Skipping {feature_dir.name}: insufficient audio files ({len(audio_files)})")
        return None

    # Split files
    gemini_files, clap_files = split_files(audio_files, gemini_percent)

    if not gemini_files or not clap_files:
        logger.warning(f"Skipping {feature_dir.name}: split resulted in empty sets")
        return None

    logger.info(f"  Using {len(gemini_files)} files for Gemini, {len(clap_files)} files for CLAP")

    try:
        # Query Gemini with concatenated audio
        audio_bytes = concatenate_and_convert_to_mp3(gemini_files)
        concept_labels = query_gemini(audio_bytes, gemini_model)
        description = concept_labels.overall_summary

        # Calculate CLAP score
        clap_score = calculate_clap_score(description, clap_files, clap_processor, clap_model, device)

        return {"feature_name": feature_dir.name, "description": description, "clap_score": clap_score}

    except Exception as e:
        logger.error(f"Error processing {feature_dir.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate music features using Gemini and CLAP")
    parser.add_argument("music_dir", type=Path, help="Directory containing subdirectories of music fragments")
    parser.add_argument(
        "--output",
        type=Path,
        default="music_feature_evaluation.csv",
        help="Output CSV file path (default: music_feature_evaluation.csv)",
    )
    parser.add_argument(
        "--gemini-model", default="gemini-1.5-flash", help="Gemini model to use (default: gemini-1.5-flash)"
    )
    parser.add_argument(
        "--clap-model", default="laion/clap-htsat-fused", help="CLAP model to use (default: laion/clap-htsat-fused)"
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device for CLAP model (default: cuda:0)")
    parser.add_argument(
        "--gemini-percent", type=float, default=80.0, help="Percentage of music used for Gemini query (default: 80.0)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Validate arguments
    if not args.music_dir.exists() or not args.music_dir.is_dir():
        raise ValueError(f"Music directory does not exist: {args.music_dir}")

    if not (0 < args.gemini_percent < 100):
        raise ValueError("Gemini percentage must be between 0 and 100")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load CLAP model
    logger.info(f"Loading CLAP model: {args.clap_model}")
    clap_processor = ClapProcessor.from_pretrained(args.clap_model)
    clap_model = ClapModel.from_pretrained(args.clap_model).to(device)
    clap_model.eval()

    # Get feature directories
    feature_dirs = [d for d in args.music_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(feature_dirs)} feature directories")

    # Process each feature directory
    results = []
    for feature_dir in feature_dirs:
        result = process_feature_directory(
            feature_dir, args.gemini_model, clap_processor, clap_model, device, args.gemini_percent
        )
        if result:
            results.append(result)

    # Write results to CSV
    if results:
        logger.info(f"Writing results to {args.output}")
        with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["feature_name", "description", "clap_score"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Evaluation complete. Processed {len(results)} features.")
    else:
        logger.warning("No results to write.")


if __name__ == "__main__":
    main()
