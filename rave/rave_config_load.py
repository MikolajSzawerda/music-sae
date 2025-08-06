import gin
import torch
from rave.model import RAVE


def loadGinConfig(config_paths: list[str]):
    gin.clear_config()
    for config_path in config_paths:
        gin.parse_config_file(config_path)


def createRaveModelFormGinConfigFile(config_paths: list[str]):
    loadGinConfig(config_paths)
    model = RAVE()
    return model


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = createRaveModelFormGinConfigFile(["configs/onnx.gin"])
    model.to(DEVICE)
    model.eval()
    print(model)


if __name__ == "__main__":
    main()
