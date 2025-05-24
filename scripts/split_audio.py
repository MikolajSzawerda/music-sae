from dataclasses import dataclass
import os

from accelerate import PartialState
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio
from demucs.pretrained import get_model
import hydra
import tqdm


def get_audio_paths(directory: str, extension=(".wav", "mp3")):
    audio_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension):
                path = os.path.join(root, file)
                audio_paths.append(path)

    return audio_paths


def separate_audio(model, device, audio_path: str):
    wav = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    ref = wav.mean(0)

    sources = apply_model(model, wav[None], device=device, num_workers=16)[0]
    sources = sources * ref.std() + ref.mean()

    vocals = None
    instruments = None

    for name, source in zip(model.sources, sources):
        if name == "vocals":
            vocals = source
        else:
            instruments = source if instruments is None else instruments + source

    return vocals, instruments


@dataclass
class SplitScriptConfig:
    audio_input_dir: str
    vocals_output_dir: str
    instruments_output_dir: str


@hydra.main(version_base=None, config_path=None)
def main(args: SplitScriptConfig):
    audio_paths = get_audio_paths(args.audio_input_dir)

    distributed_state = PartialState()

    model = get_model("htdemucs")
    model.device_map = distributed_state.device
    model.to(distributed_state.device)
    model.eval()

    save_cfg = {
        "samplerate": model.samplerate,
        "bitrate": 320,
        "clip": "rescale",
        "as_float": False,
        "bits_per_sample": 24,
    }

    os.makedirs(args.vocals_output_dir, exist_ok=True)
    os.makedirs(args.instruments_output_dir, exist_ok=True)

    with distributed_state.split_between_processes(audio_paths) as job_idxs:
        paths = (
            tqdm.tqdm(job_idxs)
            if distributed_state.is_main_process and distributed_state.local_process_index == 0
            else job_idxs
        )

        for audio_path in paths:
            vocals, instruments = separate_audio(model, distributed_state.device, audio_path)

            relative = os.path.relpath(audio_path, args.audio_input_dir)
            vocals_path = os.path.join(args.vocals_output_dir, relative)
            instruments_path = os.path.join(args.instruments_output_dir, relative)

            os.makedirs(os.path.dirname(vocals_path), exist_ok=True)
            os.makedirs(os.path.dirname(instruments_path), exist_ok=True)

            save_audio(vocals, vocals_path, **save_cfg)
            save_audio(instruments, instruments_path, **save_cfg)


if __name__ == "__main__":
    main()
