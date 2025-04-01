from rave_experiments import experiment, sae_loss, getDevice
import torch


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


def getActivation(model, training_batch):
    output = model.encode(training_batch).detach()
    output = prepareLatentTensor(model, output)
    first_decoder_layer = list(list(model.decoder.named_children())[0][1].named_children())[0][1]
    output = first_decoder_layer(output)
    return output.squeeze(-1)


def prepareTrainingHiperparams():
    hiperparams = {
        "loss": sae_loss,
        "loss_params": {"sae_diff": [], "bottlneck": [], "a_coef": 1e-3},
        "epochs": 170,
        "lr": 1e-3,
        "activation": getActivation,
    }
    return hiperparams


def main():
    experiment(
        model_path="./darbouka_onnx.ts",
        audio_folder_path="./",
        hiperparams=prepareTrainingHiperparams(),
    )


if __name__ == "__main__":
    main()
