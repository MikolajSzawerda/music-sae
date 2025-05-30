from datasets import load_dataset  # HF feature type :contentReference[oaicite:3]{index=3}
from .base import AudioDatasetPlugin
import torchaudio
from pathlib import Path
import numpy as np


def load_and_chunk_audio(audio_dir: Path, examples, model_sr=16000, chunk_duration_s=10):
    res = {"audio_tensor": [], "main_caption": []}
    for i, path in enumerate(examples["path"]):
        audio_path = audio_dir / path.replace(".mp3", ".2min.mp3")
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
            res["audio_tensor"].append(chunk_data)
            res["main_caption"].append(examples["main_caption"][i])
    return res


def load_and_chunk_vocals_and_instruments(audio_dir: Path, examples, model_sr=16000, chunk_duration_s=10):
    res = {"vocals_tensor": [], "instruments_tensor": [], "main_caption": []}

    vocals_dir = Path(str(audio_dir) + "-vocals") / "audio" / "audio"
    instruments_dir = Path(str(audio_dir) + "-instruments") / "audio" / "audio"

    for i, path in enumerate(examples["path"]):
        try:
            vocals_path = vocals_dir / path.replace(".mp3", ".2min.mp3")
            instruments_path = instruments_dir / path.replace(".mp3", ".2min.mp3")
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("ERROR WITH audio", vocals_path, e)
            continue

        vocals_tensor, sr = torchaudio.load(str(vocals_path))
        instruments_tensor, sr = torchaudio.load(str(instruments_path))

        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sr)

        vocals_resampled = transform(vocals_tensor)
        instruments_resampled = transform(instruments_tensor)

        vocals_array = vocals_resampled[0].numpy()
        instruments_array = instruments_resampled[0].numpy()

        chunk_size = model_sr * chunk_duration_s // 2
        total_samples = vocals_array.shape[0]

        for start in range(0, total_samples, chunk_size):
            end = start + chunk_size

            vocals_chunk_data = vocals_array[start:end]
            instruments_chunk_data = instruments_array[start:end]

            length = vocals_chunk_data.shape[0]
            if length < chunk_size:
                padding = np.zeros(chunk_size - length, dtype=vocals_chunk_data.dtype)
                vocals_chunk_data = np.concatenate([vocals_chunk_data, padding], axis=0)
                instruments_chunk_data = np.concatenate([instruments_chunk_data, padding], axis=0)

            res["vocals_tensor"].append(vocals_chunk_data)
            res["instruments_tensor"].append(instruments_chunk_data)
            res["main_caption"].append(examples["main_caption"][i])

    return res


class SongDescriberPlugin(AudioDatasetPlugin):
    name = "song_describer_plugin"

    def __init__(self, resample_sr: int, max_rows: int, max_pre_rows: int, seed: int, **kwargs):
        self.resample_sr = resample_sr
        self.max_rows = max_rows
        self.max_pre_rows = max_pre_rows
        self.seed = seed

    def load(
        self,
        split: str = "train",
        base_dir: Path = None,
        with_audio: bool = True,
        separate_vocals_and_instruments: bool = False,
        **kwargs,
    ):
        ds = load_dataset("csv", data_files=str(base_dir / "../song_describer.csv"), split=split)
        ds = ds.shuffle(self.seed)
        ds = ds.select_columns(["path", "caption"]).rename_column("caption", "main_caption")
        ds = ds.select(range(min(self.max_pre_rows, len(ds))))
        if with_audio:
            if separate_vocals_and_instruments:
                ds = ds.map(
                    lambda x: load_and_chunk_vocals_and_instruments(base_dir, x, model_sr=self.resample_sr),
                    batched=True,
                    batch_size=36,
                    remove_columns=["path"],
                )
            else:
                ds = ds.map(
                    lambda x: load_and_chunk_audio(base_dir / "audio" / "audio", x, model_sr=self.resample_sr),
                    batched=True,
                    batch_size=36,
                    remove_columns=["path"],
                )
        else:
            ds = ds.select_columns(["main_caption"])
        ds = ds.select(range(min(self.max_rows, len(ds))))
        return ds
