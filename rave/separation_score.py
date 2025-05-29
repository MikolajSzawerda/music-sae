import argparse
import torch
import numpy as np
import librosa
from sklearn.metrics import silhouette_score, silhouette_samples
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score


def plotExtractedConcepts(X_embedded, picthes_annotations, metric: str):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=picthes_annotations, cmap='viridis')
    plt.title(f"T-SNE: {metric}")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def annotatePitch(batches: list[np.ndarray], n_bins: int = None) -> list[float]:
    signal = np.concatenate(batches)
    frequencies = librosa.yin(signal, sr=48000, fmin=188, fmax=5000, frame_length=513, hop_length=513)
    if n_bins is not None and n_bins > 0:
        valid_freq = np.array([f for f in frequencies if not np.isnan(f) and f > 0])
        sorted_freq = np.sort(valid_freq)
        bins = np.array_split(sorted_freq, n_bins)
        bin_means = [np.mean(b) if len(b) > 0 else 0 for b in bins]
        freq_to_quantized = {}
        for b, mean in zip(bins, bin_means):
            for val in b:
                freq_to_quantized[val] = mean
        quantized_freqs = [freq_to_quantized.get(f, 0) if not np.isnan(f) and f > 0 else 0 for f in frequencies]
    else:
        quantized_freqs = frequencies
    notes = librosa.hz_to_note(quantized_freqs)
    midi_indexes = librosa.note_to_midi(notes)
    return midi_indexes


def quantize_values(values: np.ndarray, n_levels: int) -> np.ndarray:
    min_val, max_val = np.min(values), np.max(values)
    bins = np.linspace(min_val, max_val, n_levels + 1)
    quantized = np.digitize(values, bins) - 1
    quantized = np.clip(quantized, 0, n_levels - 1)
    centers = (bins[:-1] + bins[1:]) / 2
    return centers[quantized]


def annotateTempo(batches: list[np.ndarray], group_size: int = 16, quantize: bool = False, n_levels: int = 5):
    grouped_tempi = []
    n = len(batches)
    num_groups = (n + group_size - 1) // group_size
    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min((group_idx + 1) * group_size, n)
        group_audio = np.concatenate(batches[start:end])
        onset_env = librosa.onset.onset_strength(y=group_audio, sr=48000, hop_length=64)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=48000, hop_length=64)
        grouped_tempi.extend([tempo] * (end - start))
    grouped_tempi = np.array(grouped_tempi, dtype=float)
    if quantize:
        grouped_tempi = quantize_values(grouped_tempi, n_levels)
    grouped_tempi = grouped_tempi.ravel()
    return grouped_tempi.tolist()


def computeMeanActivations(encoded_tensor: torch.Tensor, labels_tensor: torch.Tensor) -> torch.Tensor:
    """
    Oblicza średnie aktywacje dla każdej unikalnej klasy w picthes_tensor.

    :param encoded_tensor: Tensor zakodowanych cech [N, D]
    :param picthes_tensor: Tensor klas pitch [N]
    :return: Tensor średnich aktywacji [C, D]
    """
    unique_classes = torch.unique(labels_tensor, return_counts=False)
    mean_activations = []
    for c in unique_classes:
        class_mask = (labels_tensor == c)
        class_features = encoded_tensor[class_mask]
        class_mean = class_features.mean(dim=0)
        mean_activations.append(class_mean)
    return torch.stack(mean_activations), unique_classes


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("encoded_path", type=str, help="Path to the file with saved tensors")
    parser.add_argument("batches_path", type=str,
                        help="Path to the file with batches corresponding to saved activation tensors")
    args = parser.parse_args()
    return args


def main():
    args = getCMDArgs()
    encoded = torch.load(args.encoded_path, weights_only=False)
    batches = torch.load(args.batches_path)
    new_batches = []
    for batch in batches:
        tensors = list(batch)
        for tensor in tensors:
            new_batches.append(tensor)
    batches = [batch.cpu().numpy().squeeze(0) for batch in new_batches]
    pitches = annotatePitch(batches, n_bins=None)
    tempos = annotateTempo(batches, group_size=16, quantize=False, n_levels=10)
    encoded_tensor = torch.tensor(encoded)
    picthes_tensor = torch.tensor(pitches)
    tempos_tensor = torch.tensor(tempos)
    topk = 10
    mean_activations, unique_classes = computeMeanActivations(encoded_tensor, picthes_tensor)
    for idx, c in enumerate(unique_classes):
        top_neurons = torch.topk(mean_activations[idx], topk).indices
        print(f"Pitch {c.item()}: Top {topk} neurons: {sorted(top_neurons.tolist())}")
    mean_activations, unique_classes = computeMeanActivations(encoded_tensor, tempos_tensor)
    for idx, c in enumerate(unique_classes):
        top_neurons = torch.topk(mean_activations[idx], topk).indices
        print(f"Tempo {c.item()}: Top {topk} neurons: {sorted(top_neurons.tolist())}")
    score = silhouette_score(encoded, pitches, metric='euclidean')
    print(f"Mean Silhouette score (Pitch): {score:.3f}")
    score = silhouette_score(encoded, tempos, metric='euclidean')
    print(f"Mean Silhouette score (Tempo): {score:.3f}")
    scores = silhouette_samples(encoded, pitches, metric="euclidean")
    print(f"% of incorrectly separated (Pitch): {len(scores[scores <=-0.1])/len(scores) * 100:.2f}")
    print(f"% of overlapping separations (Pitch):\
{len(scores[(scores > -0.1) & (scores < 0.1)])/len(scores) * 100:.2f}")
    print(f"% of well separated (Pitch): {len(scores[scores >= 0.1])/len(scores) * 100:.2f}")
    scores = silhouette_samples(encoded, tempos, metric="euclidean")
    print(f"% of incorrectly separated (Tempo): {len(scores[scores <= -0.1])/len(scores) * 100:.2f}")
    print(f"% of overlapping separations (Tempo):\
{len(scores[(scores > -0.1) & (scores < 0.1)])/len(scores) * 100:.2f}")
    print(f"% of well separated (Tempo): {len(scores[scores >= 0.1])/len(scores) * 100:.2f}")
    reg = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(reg, encoded, pitches, cv=3)
    print(f"Cross-val accuracy (Pitches): {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(reg, encoded, tempos, cv=3, scoring='r2', verbose=True)
    print(f"Cross-val R² score (Tempos): {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    X_embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(encoded)
    plotExtractedConcepts(X_embedded, pitches, "pitch")
    plotExtractedConcepts(X_embedded, tempos, "tempo")


if __name__ == "__main__":
    sys.argv = ["separation_score.py",
                "./encoded/sae_darbouka_encoder_2_Rock_1024_0/sae_darbouka_encoder_2_Rock_1024_0_id_0.npy",
                "./activations_test_batches/darbouka_encoder_2_batches_Rock/darbouka_encoder_batches_2_1024_id_0.pt"]
    main()
