import argparse
import torch
from rave_experiments import getDevice, prepareModel, AudioChunksDataset, createDataloader
import tqdm
import importlib.util
import sys
from pathlib import Path
from typing import List, Tuple
import random


def loadCallbacksModule(path: str):
    filepath = Path(path)
    module_name = filepath.stem
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    modul = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = modul
    spec.loader.exec_module(modul)
    return modul


class BadLayerNameException(Exception):
    def __init__(self):
        super.__init__("Podano nieprawidłową nazwę warstwy do zbierania aktywacji")


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", type=str, help="Path to the directory containing input for chosen \
                        trained Rave model")
    parser.add_argument("callbacks_file", type=str, help="Path to the file with necessary callbacks")
    parser.add_argument("cluster_size", type=int, help="Single activation cluster size")
    parser.add_argument("filename", type=str, help="Path to the file with chosen trained Rave model")
    parser.add_argument("layer_name", type=str, help="Layer name for gathering activations")
    parser.add_argument("output_name", type=str, help="First part of the path to the file with saved tensors")
    parser.add_argument("output_batches_name", type=str,
                        help="First part of the path to the file with saved batches corresponding to saved activations")
    parser.add_argument("--chunk_size", type=int, default=513,
                        help="Audio samples in a single tensor in an input batch for a model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of tensors in a batch")
    args = parser.parse_args()
    return args


def prepareActivationFuncParams(callback_info: dict, layer_name: str, model):
    callback_info["args"]["model"] = model
    callback_info["args"]["layer_name"] = layer_name


def chooseActivationFunction(layer_name: str, callbacks: dict):
    if layer_name not in callbacks.keys():
        raise BadLayerNameException()
    return callbacks[layer_name]


def gatherActivations(callback_info, dataloader: torch.utils.data.DataLoader, device: str):
    activations = []
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        with torch.no_grad():
            for number, batch in enumerate(dataloader):
                batch = batch.to(device)
                callback_info["args"]["batch"] = batch
                output = callback_info["callback"](**callback_info["args"])
                output = output.flatten(start_dim=1)
                activations.append((batch, output))
                pbar.set_postfix_str(f"Batch number: {number}")
                pbar.update(1)
    return activations


def getChunkList(data: List[Tuple], chunk_size: int) -> List[List[Tuple]]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def main():
    args = getCMDArgs()
    DEVICE = getDevice()
    dataset = AudioChunksDataset(args.audio_dir, chunk_size=args.chunk_size)
    dataloader = createDataloader(dataset, shuffle=True, batch_size=args.batch_size)
    model = prepareModel(args.filename, DEVICE)
    modul = loadCallbacksModule(args.callbacks_file)
    callback_info = chooseActivationFunction(args.layer_name, modul.getCallbacks())
    prepareActivationFuncParams(callback_info, args.layer_name, model)
    activations = gatherActivations(callback_info, dataloader, DEVICE)
    chunks = getChunkList(activations, chunk_size=args.cluster_size)
    random.seed(42)
    random.shuffle(chunks)
    random.seed(None)
    for iterator, chunk in enumerate(chunks):
        torch.save([activation for batch, activation in chunk], args.output_name + f"_id_{iterator}.pt")
        torch.save([batch for batch, activation in chunk], args.output_batches_name + f"_id_{iterator}.pt")


if __name__ == "__main__":
    main()
