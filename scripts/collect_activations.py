from datasets import load_dataset, Dataset
from musicsae.nnsight_model import MusicGenLanguageModel, AutoProcessor
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from transformers import set_seed
from accelerate import PartialState
from src.project_config import INPUT_DATA_DIR
import torchaudio
from datasets import Features, Array2D
from datasets.arrow_writer import ArrowWriter


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
    model_sampling_rate: int = 32000
    max_examples_per_shard: int = 100


cs = ConfigStore.instance()
cs.store(name="config", node=CollectScriptConfig)


def add_audio_to_sample(model_sr, sample):
    audio_path = INPUT_DATA_DIR / "music-bench" / "datashare" / sample["location"]
    audio_tensor, sr = torchaudio.load(str(audio_path))
    transform = torchaudio.transforms.Resample(sr, model_sr)
    sample["audio_tensor"] = transform(audio_tensor).numpy()[0]
    sample["sr"] = model_sr
    return sample


def get_batches(cfg: CollectScriptConfig) -> Dataset:
    prompts_ds = load_dataset(cfg.dataset.name, split=cfg.dataset.split).select_columns([cfg.dataset.column])
    prompts_ds = prompts_ds.shuffle(cfg.seed)
    prompts_ds = prompts_ds.select(range(cfg.dataset.max_rows))
    prompts_ds = prompts_ds.map(lambda x: add_audio_to_sample(cfg.model_sampling_rate, x)).select_columns(
        [cfg.dataset.column, "audio_tensor", "sr"]
    )
    return prompts_ds.batch(cfg.collect_batch_size)


@hydra.main(version_base=None, config_path="../conf/collect", config_name="config")
def main(args: CollectScriptConfig):
    set_seed(args.seed)
    ds = get_batches(args)
    model_name = f"facebook/musicgen-{args.model_name}"
    distributed_state = PartialState()
    with distributed_state.split_between_processes(list(args.layers)) as job_idxs:
        nn_model = MusicGenLanguageModel(model_name, device_map=args.device)
        processor = AutoProcessor.from_pretrained(model_name)
        with nn_model.generate("Hello world!", max_new_tokens=1):
            ...
        nn_model.device_map = distributed_state.device
        nn_model.to(distributed_state.device)
        for layer_id in job_idxs:
            layer = nn_model.decoder.model.decoder.layers[layer_id]
            activation_dim = nn_model.config.decoder.hidden_size
            features = Features({"activation": Array2D(shape=(activation_dim, activation_dim), dtype="float32")})
            path = (
                INPUT_DATA_DIR
                / "activation"
                / args.model_name
                / args.dataset.name.split("/")[1]
                / str(layer_id)
                / args.dataset.split
            )
            path.mkdir(exist_ok=True, parents=True)
            shard_id = 0
            example_counts = 0
            writer = ArrowWriter(features=features, path=path / f"data-{shard_id:05d}-of-99999.arrow")
            for batch in ds:
                inputs = processor(
                    audio=batch["audio_tensor"],
                    sampling_rate=32000,
                    text=batch["main_caption"],
                    padding=True,
                    return_tensors="pt",
                )
                with nn_model.trace(inputs):
                    activations = layer.output.save().view(-1, activation_dim)
                writer.write({"activation": activations})
                example_counts += activations.shape[0]
                if example_counts > args.max_examples_per_shard:
                    writer.finalize()
                    shard_id += 1
                    example_counts = 0
                    writer = ArrowWriter(features=features, path=path / f"data-{shard_id:05d}-of-99999.arrow")


if __name__ == "__main__":
    main()
