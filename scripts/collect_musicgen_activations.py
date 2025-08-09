from musicsae.nnsight_model import MusicGenLanguageModel, AutoProcessor
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers import set_seed
from accelerate import PartialState
from src.project_config import INPUT_DATA_DIR
from datasets import Features, Array2D
from datasets.arrow_writer import ArrowWriter
import tqdm
import torch
from src.dataset_plugins import get_datasets
from pathlib import Path


@dataclass
class CollectScriptConfig:
    datasets: list[dict]
    model_name: str = "small"
    device: str = "cuda"
    layers: list[int] = field(default_factory=list)
    max_gen_num_tokens: int = 255
    seed: int = 42
    collect_batch_size: int = 10
    model_sampling_rate: int = 32000
    max_examples_per_shard: int = 100
    resample_sr: int = 32000
    act_dir: str = "activation"


cs = ConfigStore.instance()
cs.store(name="config", node=CollectScriptConfig)


@hydra.main(version_base=None, config_path="../conf/collect", config_name="config")
def main(args: CollectScriptConfig):
    set_seed(args.seed)
    model_name = f"facebook/musicgen-{args.model_name}"
    distributed_state = PartialState()
    with distributed_state.split_between_processes(
        list(get_datasets(OmegaConf.to_container(args, resolve=True)))
    ) as job_idxs:
        nn_model = MusicGenLanguageModel(model_name, device_map=distributed_state.device)
        processor = AutoProcessor.from_pretrained(model_name)
        activation_dim = nn_model.config.decoder.hidden_size
        features = Features({"activation": Array2D(shape=(1, activation_dim), dtype="float32")})

        def get_writer(path: Path, shard_id: int = 0):
            return ArrowWriter(features=features, path=path / f"data-{shard_id:05d}-of-99999.arrow")

        with nn_model.generate("Hello world!", max_new_tokens=1):
            ...
        for ds_plug, ds_cfg in job_idxs:
            ds = ds_plug.prepare(**ds_cfg)
            name, split = ds_cfg["name"], ds_cfg["split"]
            gen_audio = "instruments_tensor" not in ds.column_names
            name = name if not gen_audio else f"{name}-gen"
            collect_bs = ds_cfg.get("batch_size") or args.collect_batch_size
            max_tokens = ds_cfg.get("max_tokens") or args.max_gen_num_tokens
            for layer_id in list(args.layers):
                path = INPUT_DATA_DIR / args.act_dir / args.model_name / name / str(layer_id) / split
                path.mkdir(exist_ok=True, parents=True)
                layer = nn_model.decoder.model.decoder.layers[layer_id]

                def generate_audio(batch):
                    return nn_model.collect_generation_activations(layer, batch["main_caption"], max_tokens)

                def forward_audio(batch):
                    inputs = processor(
                        audio=batch["instruments_tensor"],
                        sampling_rate=32000,
                        text=batch["main_caption"],
                        padding=True,
                        return_tensors="pt",
                    )
                    with torch.no_grad():
                        with nn_model.trace(inputs, invoker_args={"truncation": True, "max_length": max_tokens}):
                            return layer.output[0].save()

                gen_activations = generate_audio if gen_audio else forward_audio

                shard_id, example_counts, writer = 0, 0, get_writer(path)

                ds_iter = ds.iter(batch_size=collect_bs)
                for batch in (
                    tqdm.tqdm(ds_iter, total=len(ds) // collect_bs)
                    if distributed_state.is_main_process and distributed_state.local_process_index == 0
                    else ds_iter
                ):
                    activations = gen_activations(batch).view(-1, activation_dim).detach().cpu()
                    for act in activations:
                        writer.write({"activation": act.unsqueeze(0).numpy()})
                    example_counts += activations.shape[0]
                    if example_counts >= args.max_examples_per_shard:
                        writer.finalize()
                        shard_id += 1
                        example_counts, writer = 0, get_writer(path, shard_id)


if __name__ == "__main__":
    main()
