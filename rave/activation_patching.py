from activations import getDevice, AudioChunksDataset, createDataloader, prepareModel, getActivationEncoderLayer, getActivationDecoderLayer
import librosa
import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt


def gatherActivationPatch(activation_funcs_dict: dict, activation_func_params: dict, batch, activation_layers_name: str, layer_number: int, device: str):
    activation_func_params["layer_name"] = str(layer_number)
    activation_func_params["batch"] = batch
    output = activation_funcs_dict[activation_layers_name](**activation_func_params)
    batch_with_reduced_tone = np.stack([
                                    librosa.effects.pitch_shift(audio.squeeze(0).cpu().numpy(), sr=48000, n_steps=-2).reshape(1, -1)
                                    for audio in batch
                                ])
    batch_with_reduced_tone = torch.Tensor(batch_with_reduced_tone)
    batch_with_reduced_tone = batch_with_reduced_tone.to(device)
    activation_func_params["batch"] = batch_with_reduced_tone
    modified_pitch_output = activation_funcs_dict[activation_layers_name](**activation_func_params)
    return torch.norm(modified_pitch_output - output).item()


def gatherActivationPatches(activation_funcs_dict: dict, dataloader: torch.utils.data.DataLoader, model, device: str):
    activations_patches_dict = {}
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        for number, batch in enumerate(dataloader):
            batch = batch.to(device)
            activation_func_params = {"model": model, "batch": batch}
            for encoder_layer_number in range(15):
                base_name = "encoder_layer"
                if (base_name + str(encoder_layer_number) not in activations_patches_dict.keys()):
                    activations_patches_dict[base_name + str(encoder_layer_number)] = 0
                activations_patches_dict[base_name + str(encoder_layer_number)] += gatherActivationPatch(activation_funcs_dict, activation_func_params, batch, "darbouka_encoder", encoder_layer_number, device)
            for decoder_layer_number in range(9):
                base_name = "decoder_layer"
                if (base_name + str(decoder_layer_number) not in activations_patches_dict.keys()):
                    activations_patches_dict[base_name + str(decoder_layer_number)] = 0
                activations_patches_dict[base_name + str(decoder_layer_number)] += gatherActivationPatch(activation_funcs_dict, activation_func_params, batch, "darbouka_decoder", decoder_layer_number, device)
            pbar.set_postfix_str(f"Batch number: {number}")
            pbar.update(1)
    return activations_patches_dict


def plot(activation_patches_dict: dict):
    plt.bar(activation_patches_dict.keys(), activation_patches_dict.values(), color='skyblue')
    plt.xlabel("Warstwy")
    plt.ylabel("Czułość na zmianę o jeden ton w dół")
    plt.title("Czułość warstw modelu na zmianę tonu wejść")
    plt.xticks(rotation=45)
    plt.show()


def main():
    DEVICE = getDevice()
    dataset = AudioChunksDataset("./")
    dataloader = createDataloader(dataset, shuffle=True)
    model = prepareModel("darbouka_onnx.ts", DEVICE)
    activation_funcs_dict = {"darbouka_encoder": getActivationEncoderLayer,
                             "darbouka_decoder": getActivationDecoderLayer}
    activations_patches_dict = gatherActivationPatches(activation_funcs_dict, dataloader, model, DEVICE)
    plot(activations_patches_dict)


if __name__ == "__main__":
    main()