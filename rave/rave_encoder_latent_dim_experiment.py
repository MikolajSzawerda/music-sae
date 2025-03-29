from rave_experiments import experiment, sae_loss


def getActivation(model, training_batch):
    output = model.encode(training_batch).detach()
    return output.squeeze(-1)


def prepareTrainingHiperparams():
    hiperparams = {"loss": sae_loss, "loss_params": {"sae_diff": [], "bottlneck": [], "a_coef": 1e-3},
                   "epochs": 170, "lr": 1e-3, "activation": getActivation}
    return hiperparams


def main():
    experiment(model_path="./darbouka_onnx.ts", audio_folder_path="./", hiperparams=prepareTrainingHiperparams())


if __name__ == "__main__":
    main()
