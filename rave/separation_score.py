import argparse
import torch
import numpy as np
import librosa
from sklearn.metrics import silhouette_score, silhouette_samples
import sys
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE


def plotExtractedConcepts(X_embedded, picthes_annotations):
    # Wizualizacja kodów w przestrzeni 2D
    plt.figure(figsize=(10, 6))

    # Kolorowanie według pitch
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=picthes_annotations, cmap='viridis')
    plt.title("t-SNE dla kodów: Kolorowanie wg Pitch")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def annotatePitch(batches: list[np.ndarray]) -> list[float]:
    pitches = []
    for model_input_data in batches:
        frequency0 = librosa.yin(model_input_data, fmin=50, fmax=24000, sr=48000, frame_length=2048)
        note = librosa.hz_to_note(frequency0)[0][0]
        midi_index = librosa.note_to_midi(note)
        pitches.append(midi_index)
    return pitches


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
    batches = [batch.cpu().numpy() for batch in new_batches]
    pitches = annotatePitch(batches)
    score = silhouette_score(encoded, pitches, metric='euclidean')  # możesz też użyć 'cosine'
    print(f"Mean Silhouette score: {score:.3f}")
    scores = silhouette_samples(encoded, pitches, metric="euclidean")
    print(f"% of well separated: {len(scores[scores > 0])/len(scores) * 100:.2f}")
    print(f"% of incorrectly separated: {len(scores[scores < 0])/len(scores) * 100:.2f}")
    # X_embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_coded)
    # plotExtractedConcepts(X_embedded, pitches)


if __name__ == "__main__":
    sys.argv = ["separation_score.py.py",
                "./encoded/darbouka_decoder_5_encoded.pt",
                "./activations/darbouka_decoder_batches_5.pt"]
    main()
