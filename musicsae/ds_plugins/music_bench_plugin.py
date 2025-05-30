from datasets import load_dataset  # HF feature type :contentReference[oaicite:3]{index=3}
from .base import AudioDatasetPlugin
import torchaudio
from pathlib import Path


def add_audio_to_sample(audio_path: Path, model_sr, sample):
    audio_tensor, sr = torchaudio.load(str(audio_path / sample["location"]))
    transform = torchaudio.transforms.Resample(sr, model_sr)
    sample["audio_tensor"] = transform(audio_tensor).numpy()[0]
    sample["sr"] = model_sr
    return sample


def add_vocals_and_instruments_to_sample(audio_path: Path, model_sr, sample):
    vocals_tensor, sr = torchaudio.load(
        str(Path(str(audio_path) + "-vocals") / Path(str(sample["location"]).replace(".wav", ".mp3")))
    )
    instruments_tensor, sr = torchaudio.load(
        str(Path(str(audio_path) + "-instruments") / Path(str(sample["location"]).replace(".wav", ".mp3")))
    )
    transform = torchaudio.transforms.Resample(sr, model_sr)
    sample["vocals_tensor"] = transform(vocals_tensor).numpy()[0]
    sample["instruments_tensor"] = transform(instruments_tensor).numpy()[0]
    sample["sr"] = model_sr

    return sample


class MusicBenchPlugin(AudioDatasetPlugin):
    name = "music_bench_plugin"

    def __init__(self, resample_sr: int, max_rows: int, seed: int, **kwargs):
        self.resample_sr = resample_sr
        self.max_rows = max_rows
        self.seed = seed

    def load(
        self, split: str = "train", base_dir: Path = None, separate_vocals_and_instruments: bool = False, **kwargs
    ):
        ds = load_dataset("amaai-lab/MusicBench", split=split)
        ds = ds.shuffle(self.seed)
        ds = ds.select(range(min(self.max_rows, len(ds))))

        if base_dir is not None:
            if separate_vocals_and_instruments:
                ds = ds.map(
                    lambda x: add_vocals_and_instruments_to_sample(base_dir, self.resample_sr, x)
                ).select_columns(["main_caption", "vocals_tensor", "instruments_tensor", "sr"])
            else:
                ds = ds.map(lambda x: add_audio_to_sample(base_dir, self.resample_sr, x)).select_columns(
                    ["main_caption", "audio_tensor", "sr"]
                )
        else:
            ds = ds.select_columns(["main_caption"])

        return ds
