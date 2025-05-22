from dataclasses import dataclass, field
from pathlib import Path

from accelerate import PartialState
from datasets import Features, Array2D
from datasets.arrow_writer import ArrowWriter
import hydra
from nnsight import LanguageModel
from omegaconf import OmegaConf
import torch
import tqdm
from transformers import AutoModelForCausalLM, LogitsProcessorList

from src.dataset_plugins import get_datasets
from src.project_config import ACTIVATIONS_DIR

from yue.common import initialize_seed, BlockTokenRangeProcessor
from yue.yue import YuEInferenceConfig, YuEProcessorConfig, YuEProcessor


@dataclass
class CollectScriptConfig:
    datasets: list[dict]
    seed: int = 42
    model_name: str = "7B-anneal-en-cot"
    device: str = "cuda"
    layer: list[int] = field(default_factory=list)
    min_new_tokens: int = 3000
    max_new_tokens: int = 3000
    collect_batch_size: int = 10
    max_examples_per_shard: int = 100
    resample_sr: int = 32000
    inference: YuEInferenceConfig = YuEInferenceConfig()
    processor: YuEProcessorConfig = YuEProcessorConfig()


@hydra.main(version_base=None, config_path="../conf/yue-collect", config_name="config")
def main(args: CollectScriptConfig):
    initialize_seed(args.seed)

    distributed_state = PartialState()
    with distributed_state.split_between_processes(
        list(get_datasets(OmegaConf.to_container(args, resolve=True)))
    ) as job_idxs:
        device = distributed_state.device
        device = torch.device(args.device)

        processor = YuEProcessor(device, args.processor)
        model = AutoModelForCausalLM.from_pretrained(
            f"m-a-p/YuE-s1-{args.model_name}", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        model = LanguageModel(model, input_names=["input_ids"])
        model.to(device)
        model.eval()

        activation_dim = 4096
        features = Features({"activation": Array2D(shape=(1, activation_dim), dtype="float32")})

        def get_writer(path: Path, shard_id: int = 0):
            return ArrowWriter(features=features, path=path / f"data-{shard_id:05d}-of-99999.arrow")

        for ds_plug, ds_cfg in job_idxs:
            ds = ds_plug.prepare(**ds_cfg)
            name, split = ds_cfg["name"], ds_cfg["split"]
            gen_audio = "audio_tensor" not in ds.column_names
            name = name if not gen_audio else f"{name}-gen"
            collect_bs = ds_cfg.get("batch_size") or args.collect_batch_size
            max_tokens = ds_cfg.get("max_tokens") or args.max_new_tokens

            for layer_id in list(args.layers):
                path = ACTIVATIONS_DIR / "activation" / args.model_name / name / str(layer_id) / split
                path.mkdir(exist_ok=True, parents=True)

                layer = model.model.layers[layer_id]

                def generate_audio(batch):
                    print("GEN AUDIO: Error")

                def forward_audio(batch):
                    with torch.no_grad():
                        audio = torch.tensor(batch["audio_tensor"][0], dtype=torch.float32)
                        audio = audio.unsqueeze(0)

                        inputs = processor.process(batch["main_caption"][0], batch["main_caption"], audio)

                        # @TODO: 1. filter the main caption to include only popular YuE tags
                        # @TODO: 2. create lyrics for captions?

                        with model.trace(
                            inputs=inputs,
                            max_new_tokens=max_tokens,
                            min_new_tokens=args.min_new_tokens,
                            do_sample=True,
                            top_p=args.inference.top_p,
                            temperature=args.inference.temperature,
                            repetition_penalty=args.inference.repetition_penalty,
                            eos_token_id=processor.eoa,
                            pad_token_id=processor.eoa,
                            logits_processor=LogitsProcessorList(
                                [BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]
                            ),
                            guidance_scale=args.inference.guidance_scale,
                        ):
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
                        writer.write({"activation": act.unsqueeze(0).to(torch.float32).numpy()})
                    example_counts += activations.shape[0]
                    if example_counts >= args.max_examples_per_shard:
                        writer.finalize()
                        shard_id += 1
                        example_counts, writer = 0, get_writer(path, shard_id)


if __name__ == "__main__":
    main()
