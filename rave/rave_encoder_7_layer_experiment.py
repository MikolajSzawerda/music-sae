from rave_experiments import experiment, sae_loss


def getActivation(model, training_batch):
    encoder_net = list(list(list(model.encoder.named_children())[0][1].named_children())[0][1].named_children())
    output = model.pqmf(training_batch).detach()
    for name, module in encoder_net:
        output = module(output).detach()
        if name == "7":
            break
    return output.flatten(-2)


def prepareTrainingHiperparams():
    hiperparams = {
        "loss": sae_loss,
        "loss_params": {"sae_diff": [], "bottlneck": [], "a_coef": 1e-3},
        "epochs": 180,
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
