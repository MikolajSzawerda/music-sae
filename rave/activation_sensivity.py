from activations import getDevice, AudioChunksDataset, createDataloader, prepareModel
import librosa
import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import importlib.util
import sys
from pathlib import Path


def loadCallbacksModule(path: str):
    filepath = Path(path)
    module_name = filepath.stem
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    modul = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = modul
    spec.loader.exec_module(modul)
    return modul


def changePitch(batch, sr: int, pitch_modification_steps: int, device: str, n_fft: int):
    batch_with_changed_pitch = np.stack([
                                    librosa.effects.pitch_shift(audio.squeeze(0).cpu().numpy(), sr=sr,
                                                                n_steps=pitch_modification_steps,
                                                                **{"n_fft": n_fft}).reshape(1, -1)
                                    for audio in batch
                                ])
    batch_with_changed_pitch = torch.Tensor(batch_with_changed_pitch)
    batch_with_changed_pitch = batch_with_changed_pitch.to(device)
    return batch_with_changed_pitch


def changeTempo(batch, rate: float, device: str, n_fft: int):
    batch_with_changed_tempo = np.stack([
                                    librosa.util.fix_length(librosa.effects.time_stretch(audio.squeeze(0).cpu().numpy(),
                                                                                         rate=rate, **{"n_fft": n_fft}),
                                                            size=513).reshape(1, -1)
                                    for audio in batch
                                ])
    batch_with_changed_tempo = torch.Tensor(batch_with_changed_tempo)
    batch_with_changed_tempo = batch_with_changed_tempo.to(device)
    return batch_with_changed_tempo


def gatherActivationPatch(callback_info: dict,
                          batch_mod_func, batch_mod_func_args: dict):
    output = callback_info["callback"](**callback_info["args"])
    batch_mod_func_args["batch"] = callback_info["args"]["batch"]
    changed_batch = batch_mod_func(**batch_mod_func_args)
    callback_info["args"]["batch"] = changed_batch.detach()
    modified_output = callback_info["callback"](**callback_info["args"])
    return (modified_output, torch.norm(modified_output - output).item())


def gatherModuleActivations(layer_name: str,
                            activations_patches_dict: dict, tensors_pairs_dict: dict,
                            callback_info: dict,
                            batch_mod_func, batch_mod_func_args):
    if (layer_name not in activations_patches_dict.keys()):
        activations_patches_dict[layer_name] = 0
    if (layer_name not in tensors_pairs_dict.keys()):
        tensors_pairs_dict[layer_name] = []
    activation_patch = gatherActivationPatch(callback_info,
                                             batch_mod_func, batch_mod_func_args)
    tensors_pairs_dict[layer_name].append((callback_info["args"]["batch"], activation_patch[0]))
    activations_patches_dict[layer_name] += activation_patch[1]


def gatherActivationPatches(callbacks_dict: dict, dataloader: torch.utils.data.DataLoader, model, device: str,
                            batch_mod_func, batch_mod_func_args):
    activations_patches_dict = {}
    tensors_pairs_dict = {}
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        with torch.no_grad():
            for number, batch in enumerate(dataloader):
                batch = batch.to(device)
                for layer_name, callback_info in callbacks_dict.items():
                    callback_info["args"]["model"] = model
                    callback_info["args"]["batch"] = batch
                    gatherModuleActivations(layer_name,
                                            activations_patches_dict,
                                            tensors_pairs_dict, callback_info, batch_mod_func, batch_mod_func_args)
                pbar.set_postfix_str(f"Batch number: {number}")
                pbar.update(1)
    return activations_patches_dict, tensors_pairs_dict


def plot(activation_patches_dict: dict, tested_param: str, param_modification_factor, music_category: str):
    plt.figure(figsize=(19.6, 10.8))
    plt.bar(activation_patches_dict.keys(), activation_patches_dict.values(), color='skyblue')
    plt.xlabel("Layers")
    plt.ylabel(f"Sensivity to modification (normalized accumulated vector norm value): {tested_param}")
    plt.title(f"Sensivity to modification on {music_category} dataset: {tested_param}, {param_modification_factor}")
    plt.xticks(rotation=20)
    plt.savefig(f"./diagrams/Sensivity_{tested_param}_{param_modification_factor}_{music_category}.jpg")
    plt.clf()


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", type=str, help="Path to the directory containing audio input")
    parser.add_argument("callbacks_file", type=str, help="Path to a file with necessary callbacks")
    parser.add_argument("tested_parameter", type=str,
                        help="Parameter used for layers' sensivity calculation (pitch or tempo)")
    parser.add_argument("params_file", type=str, help="Path to the file containing batch modification args")
    parser.add_argument("audio_category", type=str, help="Music category")
    parser.add_argument("--chunk_size", type=int, default=513,
                        help="Audio samples in a single tensor in an input batch for a model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of tensors in a batch")
    args = parser.parse_args()
    return args


def chooseBatchModification(tested_parameter: str):
    choice_dict = {"pitch": changePitch, "tempo": changeTempo}
    return choice_dict[tested_parameter]


def chooseBatchModificationArgs(tested_param: str, params_dict: dict):
    choice_dict = {"pitch": {"sr": params_dict.get("sr", 48000),
                             "pitch_modification_steps": params_dict.get("pitch_modification_steps", -2),
                             "device": getDevice(), "n_fft": 256},
                   "tempo": {"rate": params_dict.get("rate", 0.75), "device": getDevice(), "n_fft": 256}}
    return choice_dict[tested_param]


def readParamsFile(filename: str):
    params_dict = {}
    type_map = {"int": int,
                "float": float,
                "str": str}
    with open(filename, mode="r") as file_handle:
        for line in file_handle:
            line = line.strip()
            param_name, value, param_type = line.split(",")
            params_dict[param_name] = type_map[param_type](value)
    return params_dict


def getTestedParamValueDescription(tested_param: str, params_dict: dict):
    values_dict = {"pitch": f"steps (one step is equal to semitone): \
{params_dict.get('pitch_modification_steps', -2)}",
                   "tempo": f"tempo modification scale: {params_dict.get('rate', 0.75)}"}
    return values_dict[tested_param]


def normalize(activations_sensivity_dict: dict) -> dict:
    gathered_values = []
    for layer_name in activations_sensivity_dict.keys():
        gathered_values.append(activations_sensivity_dict[layer_name])
    gathered_values = np.asarray(gathered_values)
    normalized = (gathered_values - gathered_values.min()) / (gathered_values.max() - gathered_values.min())
    for iterator, layer_name in enumerate(activations_sensivity_dict.keys()):
        activations_sensivity_dict[layer_name] = normalized[iterator]


def saveSensivityScore(filename: str, activations_sensivity_dict: dict) -> None:
    with open(filename, "w") as file_handle:
        for layer_name, sensivity_score in activations_sensivity_dict.items():
            file_handle.write(f"{layer_name},{sensivity_score}\n")


def main():
    args = getCMDArgs()
    DEVICE = getDevice()
    dataset = AudioChunksDataset(args.audio_dir, max_length=16000, chunk_size=args.chunk_size)
    dataloader = createDataloader(dataset, shuffle=True, batch_size=args.batch_size)
    model = prepareModel("darbouka_onnx.ts", DEVICE)
    modul = loadCallbacksModule(args.callbacks_file)
    callbacks = modul.getCallbacks()
    batch_mod_func = chooseBatchModification(args.tested_parameter)
    params_dict = readParamsFile(args.params_file)
    batch_mod_func_args = chooseBatchModificationArgs(args.tested_parameter, params_dict)
    activations_patches_dict, tensors_pairs_dict = gatherActivationPatches(callbacks, dataloader, model,
                                                                           DEVICE, batch_mod_func, batch_mod_func_args)
    tested_param_value_desc = getTestedParamValueDescription(args.tested_parameter, params_dict)
    torch.save(tensors_pairs_dict,
               f"./activations_sensivity_data/Sensivity_{args.tested_parameter}_{tested_param_value_desc}_\
{args.audio_category}.pt")
    normalize(activations_patches_dict)
    saveSensivityScore(f"./diagrams_data/Sensivity_{args.tested_parameter}_{tested_param_value_desc}_\
{args.audio_category}.txt", activations_patches_dict)
    plot(activations_patches_dict, args.tested_parameter, tested_param_value_desc, args.audio_category)


if __name__ == "__main__":
    main()
