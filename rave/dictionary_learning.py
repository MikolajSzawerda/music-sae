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
import json


def findPtFiles(folder_path: str) -> List[Path]:
    folder = Path(folder_path)
    return sorted(list(folder.glob("*.pt")))


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("activations_path", type=str, help="Path to the file with saved tensors")
    parser.add_argument("base_name", type=str, help="Base filename for training statistics")
    parser.add_argument("epochs", type=int, help="Number of training epochs")
    parser.add_argument("params_filepath", type=str, help="Path to the file containing training hiperparams")
    parser.add_argument("params_id", type=int, help="Hiperparams id")
    parser.add_argument("remembered_samples", type=int, help="Number of remembered samples for loss computing")
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


def makeCallback(progress_bar: tqdm):
    def callback(args):
        progress_bar.total = args["n_steps"]
        progress_bar.update(1)
        progress_bar.set_description(f"Dictionary training, Batch loss: {args['batch_cost']:.4f}")
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


def readDictParamsFile(filepath: str) -> dict[dict]:
    with open(filepath, "r") as f:
        data = json.load(f)
    params_dict = {int(k): v for k, v in data.items()}
    return params_dict


def getMiniBatchDictLearningParamsList(id: int, filepath: str) -> dict:
    to_return_params_dict = readDictParamsFile(filepath)
    return [(id, to_return_params_dict[id])]


def train(params_id: int, params_filepath: str, weights_filepath: str, epochs: int, memory_window_length: int,
          base_name: str,
          files_paths: list[str]) -> list[MiniBatchDictionaryLearning]:
    dictionaries = []
    for id, training_params in getMiniBatchDictLearningParamsList(id=params_id, filepath=params_filepath):
        train_losses = []
        val_losses = []
        components = None
        epochs_bar = tqdm(range(epochs), position=0, leave=True, dynamic_ncols=True)
        dictionaries.append(trainEpochs(training_params=training_params, params_id=params_id, files_paths=files_paths,
                                        weights_filepath=weights_filepath,
                                        train_losses=train_losses, val_losses=val_losses, components=components,
                                        epochs_bar=epochs_bar, epochs=epochs, memory_window_length=memory_window_length,
                                        base_name=base_name))
    return dictionaries


def trainEpochs(training_params: dict, params_id: int, files_paths: list[str], weights_filepath: str,
                train_losses: list[float],
                val_losses: list[float],
                components: np.ndarray | None, epochs_bar: tqdm, epochs: int,
                memory_window_length: int, base_name: str) -> MiniBatchDictionaryLearning:
    dictionary = None
    with epochs_bar:
        for epoch_number in range(epochs):
            remembered_samples, dictionary = trainEpoch(training_params=training_params, files_paths=files_paths,
                                                        weights_filepath=weights_filepath,
                                                        components=components, epochs_bar=epochs_bar,
                                                        epoch_number=epoch_number,
                                                        memory_window_length=memory_window_length)
            components = dictionary.components_
            X_train, X_val = train_test_split(remembered_samples, test_size=0.2, random_state=42)
            loss(dictionary.components_, dictionary._fit_algorithm,
                 dictionary.transform_n_nonzero_coefs, dictionary.alpha,
                 train_losses, val_losses, X_train, X_val)
            epochs_bar.update(1)
            tqdm.write(f"Epoch number: {epoch_number}, Train loss: {train_losses[-1]:.4f} Val loss:\
{val_losses[-1]:.4f}")
            saveLossPlots(train_losses, val_losses, "diagrams/" + base_name +
                          f"_loss_params_id_{params_id}.png")
            saveLossesToJson(train_losses, "train_losses", "diagrams_data/" + base_name +
                             f"_train_losses_params_id_{params_id}.json")
            saveLossesToJson(val_losses, "val_losses", "diagrams_data/" + base_name +
                             f"_val_losses_params_id_{params_id}.json")

    return dictionary


def trainEpoch(training_params: dict, files_paths: list[str], weights_filepath: str,
               components: np.ndarray | None, epochs_bar: tqdm, epoch_number: int,
               memory_window_length: int) -> tuple[np.ndarray, MiniBatchDictionaryLearning]:
    epochs_bar.set_description_str(f"Dictionary training, Epoch number {epoch_number}")
    remembered_samples = None
    batch_clusters_bar = tqdm(total=len(files_paths), position=1, leave=False, dynamic_ncols=True,
                              desc="Dictionary training over batch cluster number")
    remembered_samples, dictionary = trainBatchClusters(training_params, components=components,
                                                        batch_clusters_bar=batch_clusters_bar,
                                                        memory_window_length=memory_window_length,
                                                        remembered=remembered_samples,
                                                        files_paths=files_paths, weights_filepath=weights_filepath)
    return remembered_samples, dictionary


def trainBatchClusters(training_params: dict, components: np.ndarray | None, batch_clusters_bar: tqdm,
                       memory_window_length: int, remembered: np.ndarray | None,
                       files_paths: list[str], weights_filepath: str) -> tuple[np.ndarray, MiniBatchDictionaryLearning]:
    dictionary = None
    with batch_clusters_bar:
        for cluster_number, file_path in enumerate(files_paths):
            batch_clusters_bar.set_description_str(f"Dictionary training over batch cluster number: \
{cluster_number}")
            X = loadActivations(file_path)
            if remembered is None:
                remembered = np.ndarray(X.shape)
            n_to_add_rows = X.shape[0]
            n_rows = remembered.shape[0] + X.shape[0]
            if n_rows > memory_window_length:
                if remembered.shape[0] == memory_window_length:
                    remembered = remembered[n_to_add_rows:]
                    remembered = np.vstack([remembered, X])
                else:
                    remembered = np.vstack([remembered, X[:n_rows - memory_window_length]])
                    remembered = remembered[n_to_add_rows - (n_rows - memory_window_length):]
                    remembered = np.vstack([remembered, X[n_rows - memory_window_length:]])
            else:
                remembered = remembered = np.vstack([remembered, X])
            progress_bar = tqdm(position=2, leave=False, dynamic_ncols=True,
                                desc="Dictionary training, Batch loss")
            dictionary = MiniBatchDictionaryLearning(**training_params,
                                                     callback=makeCallback(progress_bar),
                                                     dict_init=components)
            with progress_bar:
                dictionary = dictionary.fit(X)
                progress_bar.n = progress_bar.total
            np.save(weights_filepath, dictionary.components_)
            components = dictionary.components_
            batch_clusters_bar.update(1)
    return remembered, dictionary


def encode(dictionaries: list[MiniBatchDictionaryLearning], files_paths: list[str], output_path: str) -> None:
    for dict_number, dictionary in enumerate(dictionaries):
        dictionary.callback = None
        for file_path in files_paths:
            X_coded = []
            X = loadActivations(file_path)
            for to_encode in tqdm(X, "Dictionary transform"):
                X_coded.append(dictionary.transform(to_encode.reshape(1, -1)).squeeze())
            X_coded = np.array(X_coded)
            torch.save(X_coded, output_path + file_path.stem + f'_{dict_number}_.pt')


def main():
    warnings.filterwarnings("ignore")
    args = getCMDArgs()
    files_paths = findPtFiles(args.activations_path)
    dictionaries = train(params_id=args.params_id, params_filepath=args.params_filepath,
                         weights_filepath=args.weights_path,
                         epochs=args.epochs,
                         memory_window_length=args.remembered_samples,
                         base_name=args.base_name, files_paths=files_paths)
    print("Learning finished. Starting encoding")
    for dictionary in dictionaries:
        np.save(args.weights_path, dictionary.components_)
    encode(dictionaries=dictionaries, files_paths=files_paths, output_path=args.output_path)


if __name__ == "__main__":
    main()
