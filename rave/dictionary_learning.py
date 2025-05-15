import argparse
import torch
import numpy as np
from sklearn.decomposition import DictionaryLearning
import sys


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("activations_path", type=str, help="Path to the file with saved tensors")
    parser.add_argument("batches_path", type=str,
                        help="Path to the file with batches corresponding to saved activation tensors")
    parser.add_argument("output_path", type=str, help="Path to the file with encoded tensors")
    args = parser.parse_args()
    return args


def main():
    args = getCMDArgs()
    activations = torch.load(args.activations_path)
    batches = torch.load(args.batches_path)
    new_activations = []
    for activation in activations:
        tensors = list(activation)
        for tensor in tensors:
            new_activations.append(tensor)
    new_batches = []
    for batch in batches:
        tensors = list(batch)
        for tensor in tensors:
            new_batches.append(tensor)
    max_length = 15000
    if len(new_activations) > max_length:
        new_activations = new_activations[:max_length]
    if len(new_batches) > max_length:
        new_batches = new_batches[:max_length]
    X = [activation.cpu().numpy() for activation in new_activations]
    X = np.asarray(X)
    batches = [batch.cpu().numpy() for batch in new_batches]
    batches = np.asarray(batches)
    dictionary = DictionaryLearning(n_components=256, random_state=42, verbose=True, alpha=1)
    X_coded = dictionary.fit_transform(X)
    torch.save(X_coded, args.output_path)


if __name__ == "__main__":
    sys.argv = ["dictionary_learning.py",
                "./darbouka_decoder_5.pt",
                "./darbouka_decoder_5_batches.pt",
                "darbouka_decoder_5_encoded.pt"]
    main()
