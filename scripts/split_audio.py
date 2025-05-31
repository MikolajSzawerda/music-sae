from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
import shutil

from accelerate import PartialState
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio
from demucs.pretrained import get_model
import hydra
import tqdm
import torchaudio


def search_audio_files(directory: str, extensions=(".wav", ".mp3")):
    audio_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                path = os.path.join(root, file)
                audio_paths.append(path)

    return audio_paths


def separate_audio(model, device: str, audio_path: str):
    wav = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    ref = wav.mean(0)

    sources = apply_model(model, wav[None], device=device, num_workers=1)[0]
    sources = sources * ref.std() + ref.mean()

    vocals = None
    instruments = None

    for name, source in zip(model.sources, sources):
        if name == "vocals":
            vocals = source
        else:
            instruments = source if instruments is None else instruments + source

    return vocals, instruments


def verify_audio(audio_path: str) -> bool:
    try:
        audio, _ = torchaudio.load(audio_path)
        return audio.numel() > 0
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print("Error:", e)
        return False


def do_separate_audio(model, device: str, audio_path: str, vocals_path: str, instruments_path: str, save_cfg):
    vocals, instruments = separate_audio(model, device, audio_path)

    save_audio(vocals, vocals_path, **save_cfg)
    save_audio(instruments, instruments_path, **save_cfg)


@dataclass
class SplitScriptConfig:
    audio_input_dir: str
    vocals_output_dir: str
    instruments_output_dir: str
    verify: bool = True
    num_verify_workers: int = 1


def get_output_paths(audio_path: str, args: SplitScriptConfig):
    relative = os.path.relpath(audio_path, args.audio_input_dir)
    vocals_path = os.path.join(args.vocals_output_dir, relative).replace(".wav", ".mp3")
    instruments_path = os.path.join(args.instruments_output_dir, relative).replace(".wav", ".mp3")

    return vocals_path, instruments_path


def verify_worker(args: tuple[str, SplitScriptConfig]) -> str:
    audio_path, args = args
    vocals_path, instruments_path = get_output_paths(audio_path, args)

    if not verify_audio(vocals_path) or not verify_audio(instruments_path):
        return audio_path

    return None


@hydra.main(version_base=None, config_path=None)
def main(args: SplitScriptConfig):
    audio_paths = search_audio_files(args.audio_input_dir)

    distributed_state = PartialState()
    device = distributed_state.device
    device = "cpu"

    model = get_model("htdemucs")
    model.to(device)
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
        if args.verify:
            paths = job_idxs

            with ProcessPoolExecutor(max_workers=args.num_verify_workers) as executor:
                to_regenerate = list(
                    tqdm.tqdm(
                        executor.map(
                            verify_worker,
                            [(path, args) for path in paths],
                            chunksize=16,
                        ),
                        desc="Verify",
                        total=len(paths),
                    )
                )

            to_regenerate = [path for path in to_regenerate if path is not None]

            paths = (
                tqdm.tqdm(to_regenerate, desc="Split")
                if distributed_state.is_main_process and distributed_state.local_process_index == 0
                else job_idxs
            )
        else:
            paths = (
                tqdm.tqdm(job_idxs, desc="Split")
                if distributed_state.is_main_process and distributed_state.local_process_index == 0
                else job_idxs
            )

        for audio_path in paths:
            vocals_path, instruments_path = get_output_paths(audio_path, args)

            os.makedirs(os.path.dirname(vocals_path), exist_ok=True)
            os.makedirs(os.path.dirname(instruments_path), exist_ok=True)

            max_try = 3

            for _ in range(max_try):
                do_separate_audio(model, device, audio_path, vocals_path, instruments_path, save_cfg)

                health = verify_audio(vocals_path) and verify_audio(instruments_path)

                if not args.verify:
                    return
                elif health:
                    continue

            if not health and args.verify:
                print(f"Using original audio as vocal and instruments for {audio_path}")
                shutil.copy(audio_path, vocals_path)
                shutil.copy(audio_path, instruments_path)


if __name__ == "__main__":
    main()
