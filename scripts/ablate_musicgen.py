from transformers import AutoProcessor, MusicgenForConditionalGeneration, AutoConfig
import torch
import torchaudio
from nnsight import LanguageModel
import nnsight
from IPython.display import clear_output
from musicsae.nnsight_model import MusicGenLanguageModel
from src.project_config import OUTPUT_DATA_DIR
from dataclasses import dataclass
from simple_parsing import parse, Serializable
from datasets import load_dataset
import logging
from uuid import uuid4
logger = logging.getLogger(__name__)
#
# clear_output()
# # -
#
# tokens = 255
# prompt = "Recreate the essence of a classic video game theme with chiptune sounds and nostalgic melodies."


@dataclass
class AblateScriptConfig(Serializable):
    model_name: str = 'small'
    device: str = 'cuda'
    dataset_name: str = 'amaai-lab/MusicBench'
    dataset_split: str = 'test'
    dataset_column: str = 'main_caption'
    dataset_max_rows: int = 10
    batch_size: int = 32
    music_per_prompt: int = 1
    concept: str = 'anime'
    ablation_layers: tuple[int] = (20,)
    num_tokens: int = 255

def get_prompt_batches(cfg: AblateScriptConfig):
    prompts_ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split).select_columns([cfg.dataset_column])
    prompts_ds = prompts_ds.select(range(cfg.dataset_max_rows))
    prompts_ds = prompts_ds.filter(lambda x: cfg.concept in x[cfg.dataset_column])
    logger.info(f"# of prompts: {prompts_ds.num_rows}")
    return prompts_ds.batch(cfg.batch_size // cfg.music_per_prompt)

def main():
    args = parse(AblateScriptConfig)
    nn_model = MusicGenLanguageModel(f"facebook/musicgen-{args.model_name}", device_map=args.device)
    with nn_model.generate("Hello world!", max_new_tokens=1):
        ...
    for layer_id in args.ablation_layers:
        ablate_layer = nn_model.decoder.model.decoder.layers[layer_id]
        for batch in get_prompt_batches(args):
            prompts = batch[args.dataset_column] * args.music_per_prompt
            with nn_model.generate(prompts, max_new_tokens=args.num_tokens):
                outputs = nnsight.list().save()  # Initialize & .save() nnsight list
                for _ in range(args.num_tokens):
                    ablate_layer.output[0][:] = ablate_layer.input[0][:]
                    outputs.append(nn_model.generator.output)
                    nn_model.next()
            for audio_idx in range(len(prompts)):
                path = (OUTPUT_DATA_DIR / 'ablate' / args.model_name
                        / args.dataset_name.split('/')[1]
                        / args.dataset_split / layer_id / f'out_{str(uuid4())[:6]}.wav')
                torchaudio.save(
                    path,
                    src=outputs[0][path].detach().cpu(),
                    sample_rate=nn_model.config.sampling_rate,
                    channels_first=True,
                )

if __name__ == "__main__":
    main()