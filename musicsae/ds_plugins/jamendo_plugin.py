from datasets import load_dataset  # HF feature type :contentReference[oaicite:3]{index=3}
from .base import AudioDatasetPlugin
import torchaudio
from pathlib import Path
import numpy as np


def load_and_chunk_audio(audio_dir: Path, examples, audio_col_name: str, model_sr=16000, chunk_duration_s=10):
    res = {audio_col_name: [], "main_caption": []}
    for i, path in enumerate(examples["path"]):
        audio_path = audio_dir / path

        if not audio_path.exists():
            continue

        audio_tensor, sr = torchaudio.load(str(audio_path))
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sr)
        audio_resampled = transform(audio_tensor)  # shape: (channels, samples)

        audio_array = audio_resampled[0].numpy()  # shape: (samples,)

        chunk_size = model_sr * chunk_duration_s // 2
        total_samples = audio_array.shape[0]

        for start in range(0, total_samples, chunk_size):
            end = start + chunk_size
            chunk_data = audio_array[start:end]

            length = chunk_data.shape[0]
            if length < chunk_size:
                padding = np.zeros(chunk_size - length, dtype=chunk_data.dtype)
                chunk_data = np.concatenate([chunk_data, padding], axis=0)
            res[audio_col_name].append(chunk_data)
            res["main_caption"].append(examples["main_caption"][i])
    return res


def wrapper_load_and_chunk_audio(audio_dir: Path, model_sr: int = 16000, chunk_duration_s=10):
    def wrapper(x):
        return load_and_chunk_audio(audio_dir, x, model_sr, chunk_duration_s)

    return wrapper


class JamendoPlugin(AudioDatasetPlugin):
    name = "jamendo_plugin"

    def __init__(
        self,
        resample_sr: int,
        max_rows: int,
        max_pre_rows: int,
        seed: int,
        tracks_csv: str = "tracks_filtered.csv",
        audio_col_name: str = "audio_tensor",
        **kwargs,
    ):
        self.resample_sr = resample_sr
        self.max_rows = max_rows
        self.max_pre_rows = max_pre_rows
        self.seed = seed
        self.tracks_csv = tracks_csv
        self.audio_col_name = audio_col_name

    def load(self, split: str = "train", base_dir: Path = None, with_audio: bool = True, **kwargs):
        ds = load_dataset("csv", data_files=str(base_dir / self.tracks_csv), split=split)
        ds = ds.shuffle(self.seed)
        ds = ds.select_columns(["path", "text"]).rename_column("text", "main_caption")
        ds = ds.select(range(min(self.max_pre_rows, len(ds))))

        if with_audio:
            ds = ds.map(
                lambda x: load_and_chunk_audio(
                    base_dir / "datashare-instruments",
                    x,
                    audio_col_name=self.audio_col_name,
                    model_sr=self.resample_sr,
                ),
                batched=True,
                batch_size=32,
                remove_columns=["path"],
                num_proc=1,
            )
        else:
            ds = ds.select_columns(["main_caption"])
        ds = ds.select(range(min(self.max_rows, len(ds))))
        return ds
