import json
import logging
import torchaudio
import torch
import nnsight
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable
from uuid import uuid4
from datetime import datetime

from simple_parsing import ArgumentParser
from transformers import set_seed

from dictionary_learning.trainers.top_k import AutoEncoderTopK
from musicsae.nnsight_model import MusicGenLanguageModel
from src.project_config import OUTPUT_DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


@dataclass
class InterventionConfig:
    """Configuration for SAE feature interventions."""

    # Model configuration
    model_name: str = field(
        default="facebook/musicgen-medium", metadata={"help": "HuggingFace ID for the MusicGen language model"}
    )
    ae_path: Path = field(
        default=MODELS_DIR / "medium-sae-trivial-medium-sae-ee3b/16/trainer_0/checkpoints/ae_71100.pt",
        metadata={"help": "Path to the trained AutoEncoderTopK checkpoint (*.pt)"},
    )
    layer: int = field(default=16, metadata={"help": "Decoder layer index on which to hook activations"})

    # Feature configuration
    features: List[int] = field(default_factory=list, metadata={"help": "List of feature indices to intervene on"})
    features_descriptions_path: Path = field(
        default=Path("data/input/interp/features-descriptions.json"),
        metadata={"help": "Path to JSON file containing feature descriptions"},
    )

    # Generation configuration
    max_tokens: int = field(default=255, metadata={"help": "Maximum number of tokens to generate"})
    songs_per_prompt: int = field(default=5, metadata={"help": "Number of songs to generate per prompt"})
    intervention_frequency: int = field(
        default=5, metadata={"help": "Apply intervention every N tokens (default: every 5 tokens)"}
    )

    # Intervention algorithms
    algorithm: str = field(
        default="zero_ablate",
        metadata={
            "help": "Intervention algorithm: zero_ablate, set_value, amplify, suppress, custom_negative, conditional_clamp"
        },
    )
    intervention_value: float = field(
        default=0.0, metadata={"help": "Value to use for set_value algorithm or multiplier for amplify/suppress"}
    )

    # Baseline generation options
    generate_baseline_clean: bool = field(
        default=False, metadata={"help": "Generate baseline music without any SAE intervention"}
    )
    generate_baseline_passthrough: bool = field(
        default=False, metadata={"help": "Generate baseline music with SAE encode/decode but no intervention"}
    )

    # Runtime configuration
    device: str = field(default="cuda:0", metadata={"help": "Device to run inference on"})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})

    # Output configuration
    output_dir: Path = field(
        default=OUTPUT_DATA_DIR / "sae_interventions", metadata={"help": "Output directory for generated audio files"}
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "Optional name for this run (UUID will be generated if not provided)"}
    )


class InterventionAlgorithms:
    """Collection of intervention algorithms for SAE features."""

    @staticmethod
    def create_zero_ablate(
        ae, layer, features: List[int], intervention_frequency: int, value: float = None
    ) -> Callable:
        """Zero out the specified features at regular intervals."""

        def apply_intervention(token_idx: int):
            if token_idx % intervention_frequency == 0:
                z = ae.encode(layer.output[0][:], use_threshold=True)
                for feature_idx in features:
                    z[:, :, feature_idx] = 0.0
                layer.output[0][:] = ae.decode(z)

        return apply_intervention

    @staticmethod
    def create_set_value(ae, layer, features: List[int], intervention_frequency: int, value: float) -> Callable:
        """Set the specified features to a constant value at regular intervals."""

        def apply_intervention(token_idx: int):
            if token_idx % intervention_frequency == 0 and token_idx > 0:
                z = ae.encode(layer.output[0][:], use_threshold=True)
                for feature_idx in features:
                    z[:, :, feature_idx] = value
                layer.output[0][:] = ae.decode(z)

        return apply_intervention

    @staticmethod
    def create_amplify(ae, layer, features: List[int], intervention_frequency: int, value: float) -> Callable:
        """Amplify the specified features by multiplying with value at regular intervals."""

        def apply_intervention(token_idx: int):
            if token_idx % intervention_frequency == 0 and token_idx > 0:
                z = ae.encode(layer.output[0][:], use_threshold=True)
                for feature_idx in features:
                    z[:, :, feature_idx] *= value
                layer.output[0][:] = ae.decode(z)

        return apply_intervention

    @staticmethod
    def create_suppress(ae, layer, features: List[int], intervention_frequency: int, value: float) -> Callable:
        """Suppress the specified features by dividing by value at regular intervals."""

        def apply_intervention(token_idx: int):
            if token_idx % intervention_frequency == 0:
                z = ae.encode(layer.output[0][:], use_threshold=True)
                for feature_idx in features:
                    z[:, :, feature_idx] /= value
                layer.output[0][:] = ae.decode(z)

        return apply_intervention

    @staticmethod
    def create_custom_negative(ae, layer, features: List[int], intervention_frequency: int, value: float) -> Callable:
        """Set features to negative value (like the -9 example from inter-feat.py)."""

        def apply_intervention(token_idx: int):
            if token_idx % intervention_frequency == 0:
                z = ae.encode(layer.output[0][:], use_threshold=True)
                for feature_idx in features:
                    z[:, :, feature_idx] = -abs(value)  # Ensure negative
                layer.output[0][:] = ae.decode(z)

        return apply_intervention

    @staticmethod
    def create_conditional_clamp(ae, layer, features: List[int], intervention_frequency: int, value: float) -> Callable:
        """Clamp features to maximum value when they exceed threshold."""

        def apply_intervention(token_idx: int):
            if token_idx % intervention_frequency == 0:
                z = ae.encode(layer.output[0][:], use_threshold=True)
                for feature_idx in features:
                    z[:, :, feature_idx] = torch.clamp(z[:, :, feature_idx], max=value)
                layer.output[0][:] = ae.decode(z)

        return apply_intervention

    @staticmethod
    def create_passthrough(
        ae, layer, features: List[int], intervention_frequency: int, value: float = None
    ) -> Callable:
        """Passthrough: encode and decode without intervention (for baseline comparison)."""

        def apply_intervention(token_idx: int):
            if token_idx % intervention_frequency == 0:
                z = ae.encode(layer.output[0][:], use_threshold=True)
                layer.output[0][:] = ae.decode(z)

        return apply_intervention


def load_features_descriptions(path: Path) -> Dict[str, List[str]]:
    """Load feature descriptions from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def generate_clean(
    nn_model: MusicGenLanguageModel,
    prompts: List[str],
    max_tokens: int,
) -> torch.Tensor:
    """Generate music without any SAE intervention."""
    with nn_model.generate(prompts, max_new_tokens=max_tokens):
        outputs = nnsight.list().save()

        for i in range(max_tokens):
            outputs.append(nn_model.generator.output)
            nn_model.next()

    return outputs


def generate_with_intervention(
    nn_model: MusicGenLanguageModel,
    ae: AutoEncoderTopK,
    layer,
    prompts: List[str],
    features: List[int],
    algorithm: str,
    intervention_value: float,
    max_tokens: int,
    intervention_frequency: int,
    device: str,
) -> torch.Tensor:
    """Generate music with SAE feature intervention."""

    # Get intervention function creator
    intervention_creator = getattr(InterventionAlgorithms, f"create_{algorithm}")

    # Create the intervention function with the specified parameters
    if algorithm in ["set_value", "amplify", "suppress", "custom_negative", "conditional_clamp"]:
        apply_intervention = intervention_creator(ae, layer, features, intervention_frequency, intervention_value)
    else:  # zero_ablate, passthrough don't need value
        apply_intervention = intervention_creator(ae, layer, features, intervention_frequency)

    with nn_model.generate(prompts, max_new_tokens=max_tokens):
        outputs = nnsight.list().save()

        for i in range(max_tokens):
            # Apply the intervention algorithm
            apply_intervention(i)

            outputs.append(nn_model.generator.output)
            nn_model.next()

    return outputs


def create_execution_params(config: InterventionConfig, run_id: str) -> Dict:
    """Create execution parameters for metadata."""
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "features": config.features,
        "algorithm": config.algorithm,
        "intervention_value": config.intervention_value,
        "intervention_frequency": config.intervention_frequency,
        "max_tokens": config.max_tokens,
        "songs_per_prompt": config.songs_per_prompt,
        "generate_baseline_clean": config.generate_baseline_clean,
        "generate_baseline_passthrough": config.generate_baseline_passthrough,
        "model_name": config.model_name,
        "layer": config.layer,
        "seed": config.seed,
    }


def main():
    """Main function."""
    parser = ArgumentParser()
    parser.add_arguments(InterventionConfig, dest="config")
    args = parser.parse_args()
    config: InterventionConfig = args.config

    # Set seed for reproducibility
    set_seed(config.seed)

    # Generate run ID
    run_id = config.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"

    # Setup device
    device = torch.device(config.device)

    # Load models
    logger.info(f"Loading MusicGen model: {config.model_name}")
    nn_model = MusicGenLanguageModel(config.model_name, device_map=str(device))

    logger.info(f"Loading SAE from: {config.ae_path}")
    if not config.ae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {config.ae_path}")
    ae = AutoEncoderTopK.from_pretrained(config.ae_path).to(device)

    # Get the layer to intervene on
    layer = nn_model.decoder.model.decoder.layers[config.layer]

    # Warm up the model
    with nn_model.generate("Hello world!", max_new_tokens=1):
        pass

    # Load feature descriptions
    logger.info(f"Loading feature descriptions from: {config.features_descriptions_path}")
    features_descriptions = load_features_descriptions(config.features_descriptions_path)

    # Create main run directory
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Get execution params
    execution_params = create_execution_params(config, run_id)

    # Initialize run metadata
    run_metadata = {"execution_params": execution_params, "songs": []}

    # Process each feature separately
    for feature_idx in config.features:
        feature_key = f"f{feature_idx:04d}"
        logger.info(f"Processing feature {feature_key}")

        # Create feature directory
        feature_dir = run_dir / feature_key
        feature_dir.mkdir(exist_ok=True)

        # Get prompts for this feature
        if feature_key in features_descriptions:
            prompts = features_descriptions[feature_key]
        else:
            prompts = ["A beautiful piece of music", "An emotional musical composition", "A melodic instrumental track"]
            logger.warning(f"Feature {feature_key} not found in descriptions file, using fallback prompts")

        # Generate songs for each prompt
        for prompt_idx, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {prompt_idx + 1}/{len(prompts)}: '{prompt}'")

            # Generate multiple songs for this prompt
            batch_prompts = [prompt] * config.songs_per_prompt

            generation_types = []

            # Main intervention generation
            generation_types.append((f"intervention_{config.algorithm}", config.algorithm))

            # Baseline generations if requested
            if config.generate_baseline_clean:
                generation_types.append(("musicgen", None))

            if config.generate_baseline_passthrough:
                generation_types.append(("sae_passthrough", "passthrough"))

            for algorithm_dir_name, algorithm_name in generation_types:
                try:
                    logger.info(f"Generating {config.songs_per_prompt} songs ({algorithm_dir_name})")

                    # Create algorithm directory
                    algo_dir = feature_dir / algorithm_dir_name
                    algo_dir.mkdir(exist_ok=True)

                    if algorithm_dir_name == "musicgen":
                        outputs = generate_clean(nn_model, batch_prompts, config.max_tokens)
                    elif algorithm_dir_name == "sae_passthrough":
                        outputs = generate_with_intervention(
                            nn_model=nn_model,
                            ae=ae,
                            layer=layer,
                            prompts=batch_prompts,
                            features=config.features,
                            algorithm="passthrough",
                            intervention_value=0.0,
                            max_tokens=config.max_tokens,
                            intervention_frequency=config.intervention_frequency,
                            device=str(device),
                        )
                    else:  # intervention
                        outputs = generate_with_intervention(
                            nn_model=nn_model,
                            ae=ae,
                            layer=layer,
                            prompts=batch_prompts,
                            features=[feature_idx],  # Only intervene on current feature
                            algorithm=config.algorithm,
                            intervention_value=config.intervention_value,
                            max_tokens=config.max_tokens,
                            intervention_frequency=config.intervention_frequency,
                            device=str(device),
                        )

                    # Save generated audio files
                    for song_idx in range(config.songs_per_prompt):
                        filename = f"p{prompt_idx:02d}_s{song_idx:02d}.wav"
                        output_path = algo_dir / filename

                        torchaudio.save(
                            output_path,
                            src=outputs[0][song_idx].detach().cpu(),
                            sample_rate=nn_model.config.sampling_rate,
                            channels_first=True,
                        )

                        # Add to run metadata
                        relative_path = f"{feature_key}/{algorithm_dir_name}/{filename}"
                        run_metadata["songs"].append({"filepath": relative_path, "prompt": prompt})

                    logger.info(
                        f"Saved {config.songs_per_prompt} songs ({algorithm_dir_name}) for prompt {prompt_idx + 1}/{len(prompts)}"
                    )

                except Exception as e:
                    logger.error(f"Error generating {algorithm_dir_name} music for prompt '{prompt}': {e}")
                    continue

        # Save metadata after each feature completion
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(run_metadata, f, indent=2)

        logger.info(f"Completed processing feature {feature_key} - metadata updated")

    logger.info(f"All features processed successfully! Results saved in: {run_dir}")


if __name__ == "__main__":
    main()
