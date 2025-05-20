import argparse
import torch
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import sparse_encode
from sklearn.model_selection import train_test_split
import sys
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from rave_experiments import saveLossPlots, saveLossesToJson
from typing import List


def findPtFiles(folder_path: str) -> List[Path]:
    folder = Path(folder_path)
    return sorted(list(folder.glob("*.pt")))


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("activations_path", type=str, help="Path to the file with saved tensors")
    parser.add_argument("base_name", type=str, help="Base filename for training statistics")
    parser.add_argument("output_path", type=str, help="first part of the path to the file with encoded tensors")
    parser.add_argument("weights_path", type=str, help="Path to the file with weights for the dictionary")
    args = parser.parse_args()
    return args


def loss(dictionary: np.ndarray, fit_algorithm: str, n_nonzero_coefs: int | None, alpha: float,
         train_losses: list, val_losses: list, X_train: np.ndarray, X_val: np.ndarray):
    for X, losses in [(X_train, train_losses), (X_val, val_losses)]:
        U = sparse_encode(X, dictionary=dictionary, algorithm=fit_algorithm, n_nonzero_coefs=n_nonzero_coefs,
                          alpha=alpha)
        reconstruction = U @ dictionary
        frob_loss = 0.5 * ((X - reconstruction) ** 2).sum()
        l1_loss = alpha * np.sum(np.abs(U))
        total_loss = frob_loss + l1_loss
        losses.append(total_loss)


def makeCallback(progress_bar: tqdm, batch_losses: list):
    def callback(args):
        progress_bar.total = args["n_steps"]
        # if args["i"] % args["n_steps_per_iter"] == 0:
        # loss(args["old_dict"], args["self"]._fit_algorithm,
        #      args["self"].transform_n_nonzero_coefs, args["self"].alpha,
        #      train_losses, val_losses, X_train, X_val)
        batch_losses.append(args['batch_cost'])
        progress_bar.update(1)
#         progress_bar.set_description(f"Dictionary training, Batch Loss: {args['batch_cost']} Train loss: \
# {train_losses[-1]:.4f} Val loss: {val_losses[-1]:.4f}")
        progress_bar.set_description(f"Dictionary training, Batch Loss: {args['batch_cost']:.4f}")
        progress_bar.display()
    return callback


def saveBatchLossPlot(batch_losses: list[float], filename: str):
    plt.figure(figsize=(19.2, 10.8))
    plt.plot(batch_losses, label='Batch Loss')
    plt.xlabel('Batch number')
    plt.ylabel('loss')
    plt.title('Loss over batches')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.clf()


def getMiniBatchDictLearningParamsList() -> dict[dict]:
    params_dict = {}
    params_dict[0] = {
        "n_components": 32,
        "random_state": 42,
        "verbose": False,
        "alpha": 0.5,
        "max_iter": 1000,
        "fit_algorithm": "lars",
        "n_jobs": None,
        "shuffle": True,
        "batch_size": 16,
        "tol": 0.001,
        "transform_algorithm": "omp",
        "transform_n_nonzero_coefs": None,
        "max_no_improvement": 10,
        "transform_alpha": None,
        "split_sign": False,
        "transform_max_iter": 1000
    }
    params_dict[1] = params_dict[0]
    params_dict[1]["max_iter"] = 5
    params_dict[1]["batch_size"] = 32
    to_return_params_dict = {}
    to_return_params_dict[1] = params_dict[1]
    return to_return_params_dict


def loadActivations(activations_path: str) -> np.ndarray:
    activations = torch.load(activations_path)
    new_activations = []
    for activation in activations:
        tensors = list(activation)
        for tensor in tensors:
            new_activations.append(tensor)
    X = [activation.cpu().numpy() for activation in new_activations]
    X = np.asarray(X)
    return X


def main():
    warnings.filterwarnings("ignore",
                            message="Orthogonal matching pursuit ended prematurely",
                            category=RuntimeWarning)
    args = getCMDArgs()
    files_paths = findPtFiles(args.activations_path)
    train_losses = []
    val_losses = []
    batch_losses = []
    components = None
    for id, kwargs_params in getMiniBatchDictLearningParamsList().items():
        for file_path in files_paths:
            X = loadActivations(file_path)
            X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
            progress_bar = tqdm()
            dictionary = MiniBatchDictionaryLearning(**kwargs_params,
                                                     callback=makeCallback(progress_bar, batch_losses),
                                                     dict_init=components)
            with progress_bar:
                dictionary = dictionary.fit(X_train)
                progress_bar.n = progress_bar.total
                loss(dictionary.components_, dictionary._fit_algorithm,
                     dictionary.transform_n_nonzero_coefs, dictionary.alpha,
                     train_losses, val_losses, X_train, X_val)
                progress_bar.set_description(f"Dictionary training, Batch Loss: {batch_losses[-1]:.4f} Train loss: \
{train_losses[-1]:.4f} Val loss: {val_losses[-1]:.4f}")
            components = dictionary.components_
        np.save(args.weights_path, dictionary.components_)
        saveLossPlots(train_losses, val_losses, "diagrams/" + args.base_name +
                      f"_loss_params_id_{id}.png")
        saveBatchLossPlot(batch_losses, "diagrams/" + args.base_name +
                          f"_batch_loss_params_id_{id}.png")
        saveLossesToJson(train_losses, "train_losses", "diagrams_data/" + args.base_name +
                         f"_train_losses_params_id_{id}.json")
        saveLossesToJson(val_losses, "val_losses", "diagrams_data/" + args.base_name +
                         f"_val_losses_params_id_{id}.json")
        saveLossesToJson(batch_losses, "batch_losses", "diagrams_data/" + args.base_name +
                         f"_batch_losses_params_id_{id}.json")
        print("Learning finished. Starting encoding")
        dictionary.callback = None
        for file_path in files_paths:
            X_coded = []
            X = loadActivations(file_path)
            for to_encode in tqdm(X, "Dictionary transform"):
                X_coded.append(dictionary.transform(to_encode.reshape(1, -1)).squeeze())
            X_coded = np.array(X_coded)
            torch.save(X_coded, args.output_path + file_path.stem + '.pt')


if __name__ == "__main__":
    sys.argv = ["dictionary_learning.py",
                "./activations_test",
                "darbouka_decoder_BN_7",
                "./encoded/darbouka_decoder_7_encoded_1_",
                "./weights/darbouka_decoder_7_1_dictionary_weights.npy"]
    main()
