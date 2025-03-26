import argparse
import torch
from rave_experiments import getDevice, prepareModel, AudioChunksDataset, createDataloader


class BadLayerNameException(Exception):
    def __init__(self):
        super.__init__("Podano nieprawidłową nazwę warstwy do zbierania aktywacji")


def prepareLatentTensor(model, output: torch.Tensor):
    stereo = model.stereo
    if stereo:
        z0 = torch.cat([output, output])
    else:
        z0 = output
    mode = model.mode
    if mode == "variational":
        _0 = (torch.Tensor.size(z0))[0]
        full_latent_size = model.full_latent_size
        latent_size = model.latent_size
        _1 = torch.sub(full_latent_size, latent_size)
        noise = torch.randn([_0, _1, (torch.Tensor.size(z0))[-1]])
        noise = noise.to(getDevice())
        z2 = torch.cat([z0, noise], 1)
        latent_pca = model.latent_pca
        _2 = torch.unsqueeze(latent_pca, -1)
        z3 = torch.conv1d(z2, _2)
        latent_mean = model.latent_mean
        z4 = torch.add(z3, torch.unsqueeze(latent_mean, -1))
        z1 = z4
    else:
        z1 = z0
    return z1


def getActivationEncoderLayer(model, batch, layer_name: str):
    encoder_net = list(list(list(model.encoder.named_children())[0][1].named_children())[0][1].named_children())
    output = model.pqmf(batch).detach()
    for name, module in encoder_net:
        output = module(output).detach()
        if name == layer_name:
            break
    return output


def getActivationDecoderLayer1(model, batch):
    output = model.encode(batch).detach()
    output = prepareLatentTensor(model, output)
    first_decoder_layer = list(list(model.decoder.named_children())[0][1].named_children())[0][1]
    output = first_decoder_layer(output)
    return output


def getActivationEncoder(model, batch):
    output = model.encode(batch).detach()
    return output


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", type=str, help="Path to the directory containing input for chosen \
                        trained Rave model")
    parser.add_argument("filename", type=str, help="Path to the file with chosen trained Rave model")
    parser.add_argument("layer_name", type=str, help="Layer name for gathering activations")
    parser.add_argument("output_name", type=str, help="Path to the file with saved tensors")
    args = parser.parse_args()
    return args


def prepareActivationFuncParams(layer_name: str, model):
    if layer_name in ("encoder_last", "decoder_1"):
        return {"model": model}
    return {"model": model, "layer_name": layer_name.split(sep="_")[1]}


def chooseActivationFunction(layer_name: str):
    switch = {"encoder_7": getActivationEncoderLayer,
              "encoder_last": getActivationEncoder,
              "decoder_1": getActivationDecoderLayer1}
    if layer_name not in switch.keys():
        raise BadLayerNameException()
    return switch[layer_name]


def gatherActivations(activation_func, dataloader: torch.utils.data.DataLoader, activation_func_params, device: str):
    activations = []
    for batch in dataloader:
        batch = batch.to(device)
        activation_func_params["batch"] = batch
        output = activation_func(**activation_func_params)
        output = output.flatten(start_dim=1)
        activations.append(output)
    return activations


def main():
    args = getCMDArgs()
    DEVICE = getDevice()
    dataset = AudioChunksDataset(args.audio_dir)
    dataloader = createDataloader(dataset, shuffle=True)
    model = prepareModel(args.filename, DEVICE)
    activation_function = chooseActivationFunction(args.layer_name)
    activation_function_params = prepareActivationFuncParams(args.layer_name, model)
    activations = gatherActivations(activation_function, dataloader, activation_function_params, DEVICE)
    torch.save(activations, args.output_name)


if __name__ == "__main__":
    main()
