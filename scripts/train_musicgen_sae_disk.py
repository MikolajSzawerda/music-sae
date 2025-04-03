from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer
from dictionary_learning.training import trainSAE
from datasets import load_dataset
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from src.project_config import MODELS_DIR, INPUT_DATA_DIR
import torch
from torch.utils.data import DataLoader


@dataclass
class TrainDatasetScriptConfig:
    name: str = "amaai-lab/MusicBench"
    split: str = "test"
    column: str = "main_caption"


@dataclass
class TrainScriptConfig:
    dataset: TrainDatasetScriptConfig
    model_name: str = "small"
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
    return activations.squeeze().to(device)


@hydra.main(version_base=None, config_path="../conf/musicgen-sae", config_name="config")
def main(args: TrainScriptConfig):
    for layer_id in list(args.ablation_layers):
        ds = load_dataset(
            "arrow",
            data_files=str(
                INPUT_DATA_DIR
                / "activation"
                / args.model_name
                / "MusicBench"
                / str(layer_id)
                / args.dataset.split
                / "*.arrow"
            ),
            streaming=True,
            split="train",
        )
        dl = DataLoader(
            ds, batch_size=args.activation_batch_size, num_workers=8, collate_fn=lambda x: collate_fn(x, args.device)
        )
        activation_dim = 1536
        dictionary_size = args.sae_size_multiplier * activation_dim
        trainer_cfg = {
            "trainer": TopKTrainer,
            "dict_class": AutoEncoderTopK,
            "activation_dim": activation_dim,
            "dict_size": dictionary_size,
            "lr": 1e-3,
            "device": args.device,
            "steps": args.max_steps,
            "layer": layer_id,
            "lm_name": f"musicgen-{args.model_name}",
            "warmup_steps": args.warmup_steps,
            "k": args.top_k,
            "wandb_name": str(layer_id),
        }
        trainSAE(
            data=iter(dl),
            trainer_configs=[trainer_cfg],
            steps=trainer_cfg["steps"],
            save_dir=MODELS_DIR / "musicgen-sae" / str(layer_id),
            use_wandb=True,
            wandb_project="musicgen-sae",
            log_steps=args.log_steps,
            normalize_activations=True,
        )


if __name__ == "__main__":
    main()
