import torch
from rave_experiments import prepareModel, AudioChunksDataset, createDataloader
import torchaudio
import argparse
import sys
import importlib.util
from pathlib import Path


def getCMDArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the file with chosen rave model")
    parser.add_argument("audio_dir_path", type=str, help="Path to the audio folder")
    parser.add_argument("callbacks_file", type=str, help="Path to the callbacks file with model forward functions")
    args = parser.parse_args()
    return args


def loadCallbacksModule(path: str):
    filepath = Path(path)
    module_name = filepath.stem
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    modul = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = modul
    spec.loader.exec_module(modul)
    return modul


def main():
    args = getCMDArgs()
    device = "cpu"
    model = prepareModel(args.model_path, device)
    dataset = AudioChunksDataset(args.audio_dir_path)
    dataloader = createDataloader(dataset, shuffle=False)
    waveform = None
    modul = loadCallbacksModule(args.callbacks_file)
    with torch.no_grad():
        for data in dataloader:
            output = modul.forwardWithIntervention(model, data, "0")
            audio = [audio_period for audio_period in output]
            if waveform is None:
                waveform = torch.cat(audio, dim=1)
            else:
                temp = torch.cat(audio, dim=1)
                waveform = torch.cat([waveform, temp], dim=1)
    torchaudio.save("output.wav", waveform, sample_rate=48000)


if __name__ == "__main__":
    main()
