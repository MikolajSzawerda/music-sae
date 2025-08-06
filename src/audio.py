from pathlib import Path
from typing import Any, NamedTuple
import logging

import torch
import torchaudio

from src.heatmap import (
    create_heatmap,
    get_feature_durations,
    merge_feature_durations,
    mask_heatmap_features_by_strength,
    Duration,
)


logger = logging.getLogger(__name__)


class SegmentOptions(NamedTuple):
    min_feature_strength: float = 0.1  # cells with strength below are zeroed
    max_durations_gap: int = 1  # maximum gap between two durations
    min_duration: int = 50  # minimum duration when feature is active in model units
    margin: int = 1  # margin when cutting audio
    sr: int = 32000  # sampling rate
    model_hz: int = 50  # model sample rate


class Models(NamedTuple):
    model: Any  # Nnsight model
    processor: Any  # Processor for the model
    layer: Any  # Layer of nnsight model
    ae: Any  # Autoencoder model
    device: Any  # Device for calculation
    sr: int = 32000  # Sampling rate
    max_tokens: int = 1000  # Max tokens


def load_audio_file(file_path: Path, target_sr: int = 32000) -> torch.Tensor:
    """Load and resample audio file"""

    try:
        audio_tensor, sr = torchaudio.load(str(file_path))

        if sr != target_sr:
            transform = torchaudio.transforms.Resample(sr, target_sr)
            audio_tensor = transform(audio_tensor)

        return audio_tensor.numpy()[0]  # Take first channel
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def save_audio_to_file(audio: torch.Tensor, file_path: Path, sample_rate: int = 32000) -> None:
    """Save audio to file"""
    try:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        torchaudio.save(str(file_path), audio, sample_rate)
    except Exception as e:
        logger.error(f"Error saving audio file {file_path}: {e}")
        raise


def get_audio_feature_activations(audio: torch.Tensor, models: Models) -> torch.Tensor:
    inputs = models.processor(audio=[audio], sampling_rate=models.sr, text=[""], padding=True, return_tensors="pt").to(
        models.device
    )

    with torch.no_grad():
        with models.model.trace(inputs, invoker_args={"truncation": True, "max_length": models.max_tokens}):
            act = models.layer.output[0].save()

        z = models.ae.encode(act)

    return z


def segment_audio_by_feature(
    audio: torch.Tensor, feature_index: int, models: Models, options: SegmentOptions
) -> list[torch.Tensor]:
    """Segment audio into fragments where feature is active"""

    activations = get_audio_feature_activations(audio, models)
    activations = list(activations[0])

    heatmap = create_heatmap(activations)
    heatmap = mask_heatmap_features_by_strength(heatmap, options.min_feature_strength)

    durations = get_feature_durations(heatmap[feature_index])
    durations = merge_feature_durations(durations, options.max_durations_gap)

    segments = []
    timespans = []
    for d in durations:
        if d.size() < options.min_duration:
            continue

        d = Duration(max(d.start - options.margin, 0), d.end + options.margin, d.power)
        d = d.translate(options.sr / options.model_hz)

        segments.append(audio[d.start : d.end + 1])
        timespans.append((d.start / options.sr, d.end / options.sr))

    return segments, timespans


def segment_audio_file_by_feature(
    audio_file: Path, feature_index: int, models: Models, options: SegmentOptions
) -> list[torch.Tensor]:
    """Segment audio into fragments where feature is active"""

    audio = load_audio_file(audio_file, options.sr)
    return segment_audio_by_feature(audio, feature_index, models, options)
