import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torchaudio
from datasets import load_dataset
from dotenv import load_dotenv
from simple_parsing import ArgumentParser
from tqdm import tqdm

from dictionary_learning.trainers.top_k import AutoEncoderTopK
from musicsae.nnsight_model import AutoProcessor, MusicGenLanguageModel
from src.project_config import INPUT_DATA_DIR, MODELS_DIR

BASE_DIR = INPUT_DATA_DIR / "music-bench" / "datashare-instruments"


@dataclass
class Args:
    # Core model choices
    model_name: str = field(
        default="facebook/musicgen-medium",
        metadata={"help": "HuggingFace ID for the MusicGen language model."},
    )
    ae_path: Path = field(
        default=MODELS_DIR / "medium-sae-trivial-medium-sae-ee3b/16/trainer_0/checkpoints/ae_71100.pt",
        metadata={"help": "Path to the trained AutoEncoderTopK checkpoint (*.pt)"},
    )
    layer: int = field(default=16, metadata={"help": "Decoder layer index on which to hook activations."})

    # Runtime options
    device: str = field(default="cuda:0", metadata={"help": "Torch device to run inference."})
    batch_size: int = field(default=15, metadata={"help": "Audio batch size for MusicGen."})
    max_tracks: int = field(default=10_000, metadata={"help": "Maximum number of tracks to analyse before stopping."})
    max_tokens: int = field(default=200, metadata={"help": "Language model truncation length."})

    # Output
    output: Path = field(
        default=INPUT_DATA_DIR / "interp" / "features_grouped.json",
        metadata={"help": "Destination JSON file for kept‑feature mapping."},
    )


THETA_MIN = 0.01  # ri lower bound (exclusive)
THETA_MAX = 0.25  # ri upper bound (inclusive)
ACT_THRESHOLD = 0.0  # τ – any mean activation > 0 counts as “present”
TOP_K_EXAMPLES = 10


def load_audio(base_dir: Path, rel_path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(base_dir / rel_path).replace(".wav", ".mp3"))
    return torchaudio.transforms.Resample(sr, target_sr)(wav)[0]


def process_batch(batch, base_dir: Path, model_sr: int):
    audio_tensor, caption, location = [], [], []
    for loc, cap in zip(batch["location"], batch["main_caption"]):
        if "data_aug2" in loc:
            continue  # skip augmentation copies
        try:
            audio_tensor.append(load_audio(base_dir, loc, model_sr).numpy())
        except Exception:
            continue
        caption.append(cap)
        location.append(loc)
    return {"main_caption": caption, "audio_tensor": audio_tensor, "location": location}


@torch.no_grad()
def analyse_dataset(
    ds,
    processor,
    nn_model,
    layer,
    ae,
    *,
    batch_size: int,
    max_tracks: int,
    max_tokens: int,
    device: str | torch.device,
    model_sr: int,
):
    num_features: int = ae.encoder.out_features
    sum_delta = torch.zeros(num_features, dtype=torch.float32, device=device)
    track_id_to_loc: Dict[int, str] = {}
    mean_rows, mean_index = [], []

    iterator = ds.iter(batch_size)
    global_track_id = processed_tracks = 0

    for final_batch in tqdm(iterator, desc="Analysing dataset"):
        batch = process_batch(final_batch, BASE_DIR, model_sr)
        B = len(batch["audio_tensor"])
        if B == 0:
            continue
        track_ids = torch.arange(global_track_id, global_track_id + B)
        global_track_id += B
        for i in range(B):
            track_id_to_loc[int(track_ids[i])] = batch["location"][i]

        inputs = processor(
            audio=batch["audio_tensor"],
            sampling_rate=model_sr,
            text=batch["main_caption"],
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            with nn_model.trace(inputs, invoker_args={"truncation": True, "max_length": max_tokens}):
                act = layer.output[0].save()
            z = ae.encode(act)

        batch_mean_act = z.mean(dim=1)
        mean_rows.append(batch_mean_act.cpu())
        mean_index.extend(track_ids.tolist())

        delta = batch_mean_act > ACT_THRESHOLD  # (B, F) bool mask
        sum_delta += delta.float().sum(dim=0)

        processed_tracks += B
        if processed_tracks >= max_tracks:
            break

    mean_tensor = torch.cat(mean_rows, dim=0)
    feature_cols = [f"f{idx:04d}" for idx in range(num_features)]
    mean_df = pd.DataFrame(mean_tensor.numpy(), index=mean_index, columns=feature_cols)

    n_tracks = len(mean_df)
    r_i = (sum_delta.detach().cpu() / n_tracks).numpy()
    corpus_df = pd.DataFrame({"feature": feature_cols, "activation_rate": r_i})
    corpus_df["kept"] = (corpus_df.activation_rate > THETA_MIN) & (corpus_df.activation_rate <= THETA_MAX)

    kept_features = corpus_df[corpus_df.kept].feature.tolist()
    tracks_per_feat: Dict[str, List[str]] = {}
    for feat_name in kept_features:
        non_zero_mask = mean_df[feat_name] != 0
        present_tracks = mean_df[non_zero_mask][feat_name]
        top_ids = present_tracks.nlargest(TOP_K_EXAMPLES).index.tolist()
        tracks_per_feat[feat_name] = [track_id_to_loc[tid] for tid in top_ids]

    return corpus_df, tracks_per_feat


def main() -> None:
    load_dotenv()
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args

    model_sr = 32_000  # MusicGen models operate at 32 kHz

    device = torch.device(args.device)

    nn_model = MusicGenLanguageModel(args.model_name, device_map=str(device))
    processor = AutoProcessor.from_pretrained(args.model_name)

    if not args.ae_path.is_file():
        raise FileNotFoundError(f"SAE checkpoint not found: {args.ae_path}")
    ae = AutoEncoderTopK.from_pretrained(args.ae_path).to(device)
    layer = nn_model.decoder.model.decoder.layers[args.layer]

    ds = load_dataset("amaai-lab/MusicBench", split="train", streaming=True)

    _, kept = analyse_dataset(
        ds,
        processor,
        nn_model,
        layer,
        ae,
        batch_size=args.batch_size,
        max_tracks=args.max_tracks,
        max_tokens=args.max_tokens,
        device=device,
        model_sr=model_sr,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump({k: list(set(v)) for k, v in kept.items()}, fh, indent=2)

    print(f"✓ Saved kept‑feature mapping to {args.output.resolve()}")


if __name__ == "__main__":
    main()
