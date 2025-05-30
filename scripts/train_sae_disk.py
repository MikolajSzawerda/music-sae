from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer
from dictionary_learning.training import trainSAE
from datasets import load_dataset
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from src.project_config import MODELS_DIR, INPUT_DATA_DIR
import torch
from torch.utils.data import DataLoader
from uuid import uuid4


@dataclass
class TrainDatasetScriptConfig:
    regex_name: str = "*"
    split: str = "train"


@dataclass
class TrainScriptConfig:
    project_name: str
    dataset: TrainDatasetScriptConfig
    model_name: str = "small"
    device: str = "cuda"
    ablation_layers: list[int] = field(default_factory=list)
    max_steps: int = 1000
    activation_batch_size: int = 10
    warmup_steps: int = 10
    top_k: int = 5
    seed: int = 42
    sae_size_multiplier: int = 16
    log_steps: int = 30
    save_steps: int | None = None
    auxk_alpha: float = 0.03125
    lr: float = 1e-3
    n_cpu_workers: int = 12


cs = ConfigStore.instance()
cs.store(name="config", node=TrainScriptConfig)
cs.store(name="config-yue", node=TrainScriptConfig)


def collate_fn(batch, device):
    activations = torch.tensor([item["activation"] for item in batch])
    return activations.squeeze()


@hydra.main(version_base=None, config_path="../conf/sae", config_name="config")
def main(args: TrainScriptConfig):
    for layer_id in list(args.ablation_layers):
        ds = load_dataset(
            "arrow",
            data_files=str(
                INPUT_DATA_DIR
                / "activation"
                / args.model_name
                / args.dataset.regex_name
                / str(layer_id)
                / args.dataset.split
                / "*.arrow"
            ),
            streaming=True,
            split="train",
        )
        dl = DataLoader(
            ds,
            batch_size=args.activation_batch_size,
            num_workers=min(ds.n_shards, args.n_cpu_workers),
            collate_fn=lambda x: collate_fn(x, args.device),
        )
        activation_dim = ds.features["activation"].shape[1]
        dictionary_size = args.sae_size_multiplier * activation_dim
        trainer_cfg = {
            "trainer": TopKTrainer,
            "dict_class": AutoEncoderTopK,
            "activation_dim": activation_dim,
            "dict_size": dictionary_size,
            "lr": args.lr,
            "device": args.device,
            "steps": args.max_steps,
            "layer": layer_id,
            "lm_name": f"{args.model_name}",
            "warmup_steps": args.warmup_steps,
            "k": args.top_k,
            "wandb_name": str(layer_id),
            "auxk_alpha": args.auxk_alpha,
        }
        if args.save_steps:
            save_steps = list(range(args.save_steps, args.max_steps, args.save_steps))
        else:
            save_steps = []
        run_name = f"{args.project_name}-{args.model_name}-sae-{str(uuid4())[:4]}"
        run_cfg = {
            "batch_size": args.activation_batch_size,
            "expansion_factor": args.sae_size_multiplier,
            "save_steps": save_steps,
            "run_name": run_name,
        }

        def inf_iter(loader):
            while True:
                it = iter(loader)
                for res in it:
                    yield res

        trainSAE(
            data=inf_iter(dl),
            trainer_configs=[trainer_cfg],
            steps=trainer_cfg["steps"],
            save_dir=MODELS_DIR / run_name / str(layer_id),
            use_wandb=True,
            wandb_project=args.project_name,
            log_steps=args.log_steps,
            normalize_activations=False,
            run_cfg=run_cfg,
            device=args.device,
        )


if __name__ == "__main__":
    main()
