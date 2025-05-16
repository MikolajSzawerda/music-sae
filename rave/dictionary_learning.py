import argparse
import torch
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
import sys
from tqdm import tqdm
import warnings


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("activations_path", type=str, help="Path to the file with saved tensors")
    parser.add_argument("output_path", type=str, help="Path to the file with encoded tensors")
    parser.add_argument("weights_path", type=str, help="Path to the file with weights for the dictionary")
    args = parser.parse_args()
    return args


def makeCallback(progress_bar: tqdm):
    def callback(args):
        progress_bar.update(1)
        progress_bar.display()
        if progress_bar.n >= progress_bar.total:
            progress_bar.reset()
    return callback


def main():
    args = getCMDArgs()
    activations = torch.load(args.activations_path)
    new_activations = []
    for activation in activations:
        tensors = list(activation)
        for tensor in tensors:
            new_activations.append(tensor)
    # max_length = 10000
    # if len(new_activations) > max_length:
    #     new_activations = new_activations[:max_length]
    X = [activation.cpu().numpy() for activation in new_activations]
    X = np.asarray(X)
    progress_bar = tqdm(total=1000, desc="Dictionary training")
    dictionary = MiniBatchDictionaryLearning(n_components=2048, random_state=42, verbose=False, alpha=1,
                                             max_iter=1000, fit_algorithm="lars", n_jobs=None,
                                             callback=makeCallback(progress_bar), shuffle=True, batch_size=32,
                                             tol=0.001, transform_algorithm="omp", transform_n_nonzero_coefs=None,
                                             max_no_improvement=10, transform_alpha=None, split_sign=False,
                                             transform_max_iter=1000)
    with progress_bar:
        dictionary = dictionary.fit(X)
        progress_bar.n = progress_bar.total
    np.save(args.weights_path, dictionary.components_)
    print("Learning finished. Starting encoding")
    X_coded = []
    warnings.filterwarnings("ignore",
                            message="Orthogonal matching pursuit ended prematurely",
                            category=RuntimeWarning)
    for to_encode in tqdm(X, "Dictionary transform"):
        X_coded.append(dictionary.transform(to_encode.reshape(1, -1)).squeeze())
    X_coded = np.array(X_coded)
    torch.save(X_coded, args.output_path)


if __name__ == "__main__":
    sys.argv = ["dictionary_learning.py",
                "./activations/darbouka_decoder_5_BN.pt",
                "./encoded/darbouka_decoder_5_encoded_BN_2048.pt",
                "./weights/darbouka_decoder_5_BN_2048_dictionary_weights.npy"]
    main()
