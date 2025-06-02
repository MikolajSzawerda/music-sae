import io
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import torch
import torchaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from simple_parsing import ArgumentParser

from src.project_config import INPUT_DATA_DIR

# ------------------------------- Response schema ---------------------------- #


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


PROMPT = """
Listen very carefully to this set of audio clips, which consists of song snippets concatenated in random order. You need to discover common musical patterns across the whole set, to identify what musical feature is shared across all clips. You will need to listen carefully. For each potential concept you identify, output a name, a confidence score between 0 and 1 (where 1 is highest confidence), and a concise description of the concept.
At a higher level, describe the overall concept shared across the set, give it a suitable name, and provide an overall confidence score (0 to 1).
Describe the **underlying concepts** not the specific audio snippets (e.g. your description could say "the concept" but not "the audio snippets"). However, try to avoid such verbiage altogether and concisely describe the musical concept’s main attributes.
Include NO FILLER text.
Focus on being specific. Concepts could relate to genre (e.g., hip-hop, salsa, reggaeton, balkan), instruments (e.g., piano, cello, guitar, flute), recording/production techniques (e.g., reverberation, drones, noise, DJ scratching, beatboxing, drum machine, hi-hat patterns, fingerpicking, live recording artifacts, low-pass filtering), or more nuanced musical ideas (e.g., drum solo, chill dance rhythm, serene woodwinds arrangement). These are illustrative examples, NOT a fixed list to choose from.
""".strip()


# ----------------------------- Argument parsing ----------------------------- #


@dataclass
class Args:
    """Command‑line arguments."""

    model: str = field(
        default="gemini-1.5-flash",
        metadata={"help": "Google GenAI model name (e.g. gemini-1.5-flash)"},
    )
    num_features: int = field(
        default=1,
        metadata={
            "help": (
                "Number of feature keys to process from features.json. "
                "If greater than available, all features are processed."
            )
        },
    )
    output: Path = field(
        default=Path("concept_labels_results.json"),
        metadata={"help": "File to write incremental JSON results."},
    )
    random: bool = field(
        default=False,
        metadata={"help": "Randomly choose feature keys instead of the first N."},
    )


# ----------------------------- Helper functions ----------------------------- #


def load_feature_keys(num_features: int, *, random_choice: bool = False) -> tuple[list[str], dict[str, list[str]]]:
    """Load the feature list and return the keys to process and the full dict."""

    features_path = INPUT_DATA_DIR / "interp" / "features.json"
    with features_path.open("r") as fh:
        features_dict: Dict[str, List[str]] = json.load(fh)

    keys = list(features_dict.keys())
    if random_choice:
        random.shuffle(keys)

    return keys[: min(num_features, len(keys))], features_dict


def concatenate_audio(paths: Set[str], base_dir: Path) -> bytes:
    """Load, resample and concatenate audio files specified by paths into MP3 bytes."""

    waveforms: List[torch.Tensor] = []
    for rel_path in paths:
        path = str(base_dir / rel_path).replace(".wav", ".mp3")
        tensor, sr = torchaudio.load(path)
        resampled = torchaudio.transforms.Resample(sr, 32_000)(tensor)[0]  # mono
        waveforms.append(resampled)

    concatenated = torch.cat(waveforms, dim=0).unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, concatenated, 32_000, format="mp3")
    return buf.getvalue()


# ------------------------------- Main routine -------------------------------- #


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args

    load_dotenv()

    # Load any existing results so we can resume incremental runs
    results: Dict[str, Dict] = {}
    if args.output.exists():
        with args.output.open("r") as fh:
            results = json.load(fh)

    feature_keys, all_features = load_feature_keys(args.num_features, random_choice=args.random)

    base_dir = INPUT_DATA_DIR / "music-bench" / "datashare-instruments"

    # Initialise Google GenAI client
    client = genai.Client(api_key=os.getenv("GENMINI_API"))

    for key in feature_keys:
        if key in results:
            print(f"Skipping '{key}' (already processed).")
            continue

        audio_bytes = concatenate_audio(set(all_features[key]), base_dir)

        response = client.models.generate_content(
            model=args.model,
            contents=[
                PROMPT,
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": ConceptLabels,
            },
        )

        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError as exc:
            print(f"Failed to parse JSON for '{key}': {exc}")
            continue

        results[key] = parsed

        # Persist after each key so partial runs are saved
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Processed '{key}' (total saved: {len(results)}/{args.num_features}).")

    print(f"All done! Results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
