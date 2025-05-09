import torch
from rave_experiments import prepareModel, AudioChunksDataset, createDataloader
import torchaudio


def main():
    device = "cpu"
    model = prepareModel("./darbouka_onnx.ts", device)
    dataset = AudioChunksDataset("./")
    dataloader = createDataloader(dataset, shuffle=False)
    with torch.no_grad():
        for data in dataloader:
            output = model(data)
            audio = [audio_period for audio_period in output]
            waveform = audio[0]
            waveform = torch.cat(audio[1:], dim=1)
    torchaudio.save("output.wav", waveform, sample_rate=48000)


if __name__ == "__main__":
    main()
