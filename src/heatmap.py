import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns


FeatureIndex = int
FeatureStrength = float
HeatMap = dict[FeatureIndex, list[FeatureStrength]]


class Duration:
    def __init__(self, start: int, end: int, power: float = 0):
        self.start = start
        self.end = end
        self.power = power

    def size(self):
        return self.end - self.start + 1

    def translate(self, scale: float) -> "Duration":
        return Duration(math.floor(self.start * scale), math.ceil(self.end * scale), self.power)


def create_heatmap(activations: torch.Tensor, eps: float = 0.001) -> HeatMap:
    """Create heatmap for feature activations"""
    heatmap = dict()

    for idx, act in enumerate(activations):
        active_features = (abs(act) > eps).nonzero(as_tuple=False)

        for feature in active_features:
            feature = feature.item()

            if feature not in heatmap:
                # For new features, fill previous slots in heatmap with zeros
                heatmap[feature] = [0] * idx

            # Fill the heatmap slot with feature strenght
            heatmap[feature].append(act[feature].item())

        for feature in heatmap.keys():
            if len(heatmap[feature]) != idx + 1:
                # For all inactive features in this tick, fill them with zeros
                heatmap[feature].append(0)

    return heatmap


def get_feature_durations(values: list[float]) -> list[Duration]:
    """Get all durations when feature was active"""
    durations = []

    begin = None
    power = 0
    for idx, value in enumerate(values):
        if value != 0:
            if begin is None:
                begin = idx

            power += value
        else:
            if begin is not None:
                durations.append(Duration(begin, idx - 1, power))
                begin = None
                power = 0

    if begin is not None:
        durations.append(Duration(begin, len(values), power))

    # Calculate average power for duration
    durations = [Duration(d.start, d.end, d.power / d.size()) for d in durations]

    return durations


def merge_feature_durations(durations: list[Duration], max_gap: int) -> list[Duration]:
    """Merge durations if the gap between them is within max_gap"""

    if len(durations) < 2:
        return durations

    durations.sort(key=lambda d: d.start)

    result = []
    current = durations[0]

    for upcoming in durations[1:]:
        gap = upcoming.start - current.end - 1

        if gap <= max_gap:
            # Merge current and upcoming
            average_power = (current.size() * current.power + upcoming.size() * upcoming.power) / (
                current.size() + upcoming.size()
            )

            current = Duration(start=current.start, end=upcoming.end, power=average_power)
        else:
            # Gap too large
            result.append(current)
            current = upcoming

    result.append(current)
    return result


def mask_heatmap_features_by_strength(heatmap: HeatMap, threshold: float) -> HeatMap:
    """Mask all feature activations that do not meet the threshold by setting them to zero."""
    return {key: [value if value >= threshold else 0 for value in values] for key, values in heatmap.items()}


def filter_heatmap_min_duration(heatmap: HeatMap, min_duration: int) -> HeatMap:
    """Filter features with continous duration at least of some duration"""
    result = dict()

    for key in heatmap.keys():
        durations = get_feature_durations(heatmap[key])

        for duration in durations:
            if duration.size() >= min_duration:
                result[key] = heatmap[key]
                break

    return result


def filter_heatmap_top_k_features(heatmap: HeatMap, k: int) -> HeatMap:
    """Filter top k features with highest average power"""
    result = dict()

    power_map = {key: sum(value) / len(value) for key, value in heatmap.items()}

    for key in list(reversed(sorted(power_map.keys(), key=lambda x: power_map[x])))[:k]:
        result[key] = heatmap[key]

    return result


def plot_heatmap(heatmap: HeatMap):
    """Plot the heatmap"""
    sorted_keys = list(sorted(heatmap.keys()))
    sorted_data = np.array([heatmap[key] for key in sorted_keys])

    width = 4 if len(heatmap.keys()) == 0 else len(list(heatmap.values())[0]) // 8 + 2
    height = len(heatmap.keys()) // 10 + 2
    cmap = sns.light_palette("blue", as_cmap=True)

    plt.figure(figsize=(width, height))
    sns.heatmap(sorted_data, cmap=cmap, vmin=0, vmax=1, yticklabels=sorted_keys, cbar=False)

    plt.title("Heatmap")
    plt.xlabel("Tick")
    plt.ylabel("Feature Index")
    plt.tight_layout()

    return plt


# @TODO: Function to make activations from the audio

# @TODO: Function to cut the audio into segments by duration
