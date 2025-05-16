from musicsae.usae import USAETopKTrainer, UniversalAutoEncoder
from train_usae import trainSAE
from datasets import load_dataset
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from src.project_config import MODELS_DIR, INPUT_DATA_DIR
import torch
from torch.utils.data import DataLoader
from typing import Dict, Iterator


@dataclass
class TrainDatasetScriptConfig:
    name: str = "amaai-lab/MusicBench"
    split: str = "test"
    column: str = "main_caption"


@dataclass
class TrainScriptConfig:
    dataset: TrainDatasetScriptConfig
    model_names: list[str] = field(default_factory=list)
    activation_dims: list[dict] = field(default_factory=list)
    device: str = "cuda"
    ablation_layers: list[int] = field(default_factory=list)
    max_gen_num_tokens: int = 255
    max_steps: int = 1000
    activation_buffer_size: int = 100
    activation_batch_size: int = 10
    warmup_steps: int = 10
    top_k: int = 5
    seed: int = 42
    sae_size_multiplier: int = 16
    text_batch_size: int = 10
    log_steps: int = 30


cs = ConfigStore.instance()
cs.store(name="config", node=TrainScriptConfig)


def collate_fn(batch, device):
    activations = torch.tensor([item["activation"] for item in batch])
    return activations.squeeze()


def combined_iter(dataloaders: Dict[str, DataLoader]) -> Iterator[Dict[str, dict]]:
    names = list(dataloaders.keys())
    for batches in zip(*(dataloaders[name] for name in names)):
        yield {name: batch for name, batch in zip(names, batches)}


@hydra.main(version_base=None, config_path="../conf/musicgen-usae", config_name="config")
def main(args: TrainScriptConfig):
    for layer_id in list(args.ablation_layers):
        loaders: Dict[str, DataLoader] = {}
        for model_name in args.model_names:
            ds = load_dataset(
                "arrow",
                data_files=str(
                    INPUT_DATA_DIR
                    / "activation"
                    / model_name
                    / "*_plugin"
                    / str(layer_id)
                    / args.dataset.split
                    / "*.arrow"
                ),
                streaming=True,
                split="train",
            )
            loaders[model_name] = DataLoader(
                ds,
                batch_size=args.activation_batch_size,
                num_workers=8,
                collate_fn=lambda x: collate_fn(x, args.device),
            )
        dims = {k: v for d in args.activation_dims for k, v in d.items()}
        dictionary_size = args.sae_size_multiplier * max(dims.values())
        trainer_cfg = {
            "trainer": USAETopKTrainer,
            "dict_class": UniversalAutoEncoder,
            "activation_dims": dims,
            "dict_size": dictionary_size,
            "lr": 1e-3,
            "device": args.device,
            "steps": args.max_steps,
            "layer": layer_id,
            "lm_name": f"musicgen-{'-'.join(args.model_names)}",
            "warmup_steps": args.warmup_steps,
            "k": args.top_k,
            "wandb_name": str(layer_id),
        }
        trainSAE(
            data=combined_iter(loaders),
            trainer_configs=[trainer_cfg],
            steps=trainer_cfg["steps"],
            save_dir=MODELS_DIR / "musicgen-usae" / str(layer_id),
            use_wandb=True,
            wandb_project="musicgen-usae",
            log_steps=args.log_steps,
            normalize_activations=True,
        )


if __name__ == "__main__":
    main()
