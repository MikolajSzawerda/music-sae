from datasets import load_dataset, Dataset
from musicsae.nnsight_model import MusicGenLanguageModel
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from transformers import set_seed
from accelerate import PartialState


@dataclass
class CollectDatasetScriptConfig:
    name: str = "amaai-lab/MusicBench"
    split: str = "test"
    column: str = "main_caption"
    max_rows: int = 100


@dataclass
class CollectScriptConfig:
    dataset: CollectDatasetScriptConfig
    model_name: str = "small"
    device: str = "cuda"
    layers: list[int] = field(default_factory=list)
    max_gen_num_tokens: int = 255
    seed: int = 42
    collect_batch_size: int = 10


cs = ConfigStore.instance()
cs.store(name="config", node=CollectScriptConfig)


def get_batches(cfg: CollectScriptConfig) -> Dataset:
    prompts_ds = load_dataset(cfg.dataset.name, split=cfg.dataset.split).select_columns([cfg.dataset.column])
    prompts_ds = prompts_ds.shuffle(cfg.seed)
    prompts_ds = prompts_ds.select(range(cfg.dataset.max_rows))
    return prompts_ds.batch(cfg.collect_batch_size)


@hydra.main(version_base=None, config_path="../conf/collect", config_name="config")
def main(args: CollectScriptConfig):
    set_seed(args.seed)
    # ds = get_batches(args)
    distributed_state = PartialState()

    with distributed_state.split_between_processes(list(args.layers)):
        nn_model = MusicGenLanguageModel(f"facebook/musicgen-{args.model_name}", device_map=args.device)
        with nn_model.generate("Hello world!", max_new_tokens=1):
            ...
        nn_model.device_map = distributed_state.device
        nn_model.to(distributed_state.device)
        # for layer_id in job_idxs:
        #     layer = nn_model.decoder.model.decoder.layers[layer_id]
        #     activation_dim = nn_model.config.decoder.hidden_size  # output dimension of the MLP
        #     for prompts in ds:
        #         activations = nn_model.collect_generation_activations(layer, prompts, args.max_gen_num_tokens)


if __name__ == "__main__":
    main()
