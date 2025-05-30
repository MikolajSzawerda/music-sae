from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer
from dictionary_learning.training import trainSAE
from musicsae.nnsight_model import MusicActivationBuffer
from datasets import load_dataset, Dataset
from musicsae.nnsight_model import MusicGenLanguageModel
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from src.project_config import MODELS_DIR


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


def get_ds(cfg: TrainScriptConfig) -> Dataset:
    prompts_ds = load_dataset(cfg.dataset.name, split=cfg.dataset.split).select_columns([cfg.dataset.column])
    prompts_ds = prompts_ds.shuffle(cfg.seed)
    return prompts_ds


@hydra.main(version_base=None, config_path="../conf/musicgen-sae", config_name="config")
def main(args: TrainScriptConfig):
    # set_seed(args.seed)
    if args.device == "cuda":
        sae_device, data_device = "cuda:0", "cuda:1"
    else:
        sae_device, data_device = args.device, args.device
    ds = get_ds(args)
    for layer_id in list(args.ablation_layers):
        nn_model = MusicGenLanguageModel(f"facebook/musicgen-{args.model_name}", device_map=sae_device)
        with nn_model.generate("Hello world!", max_new_tokens=1):
            ...
        nn_model.device_map = data_device
        nn_model.to(data_device)
        layer = nn_model.decoder.model.decoder.layers[layer_id]
        activation_dim = nn_model.config.decoder.hidden_size  # output dimension of the MLP
        dictionary_size = args.sae_size_multiplier * activation_dim
        buffer = MusicActivationBuffer(
            data=ds,
            data_column=args.dataset.column,
            model=nn_model,
            submodule=layer,
            d_submodule=activation_dim,
            n_ctxs=args.activation_buffer_size // args.max_gen_num_tokens,
            ctx_len=args.max_gen_num_tokens,
            refresh_batch_size=args.text_batch_size,
            out_batch_size=args.activation_batch_size,
            device=data_device,
            sae_device=sae_device,
        )
        buffer.refresh()
        trainer_cfg = {
            "trainer": TopKTrainer,
            "dict_class": AutoEncoderTopK,
            "activation_dim": activation_dim,
            "dict_size": dictionary_size,
            "lr": 1e-3,
            "device": sae_device,
            "steps": args.max_steps,
            "layer": layer_id,
            "lm_name": f"musicgen-{args.model_name}",
            "warmup_steps": args.warmup_steps,
            "k": args.top_k,
            "wandb_name": str(layer_id),
        }
        trainSAE(
            data=buffer,
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
