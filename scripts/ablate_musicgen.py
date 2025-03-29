import torchaudio
import nnsight
from musicsae.nnsight_model import MusicGenLanguageModel
from src.project_config import OUTPUT_DATA_DIR
from dataclasses import dataclass
from simple_parsing import parse, Serializable, list_field
from datasets import load_dataset
import logging
from uuid import uuid4
from tqdm import tqdm
from accelerate import PartialState
logger = logging.getLogger(__name__)


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
    ablation_layers: list[int] = list_field()
    num_tokens: int = 255

def get_prompt_batches(cfg: AblateScriptConfig):
    prompts_ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split).select_columns([cfg.dataset_column])
    prompts_ds = prompts_ds.filter(lambda x: cfg.concept in x[cfg.dataset_column])
    prompts_ds = prompts_ds.select(range(cfg.dataset_max_rows))
    logger.info(f"# of prompts: {prompts_ds.num_rows}")
    return prompts_ds.batch(cfg.batch_size // cfg.music_per_prompt)

def main():
    args = parse(AblateScriptConfig)
    batches = get_prompt_batches(args)
    nn_model = MusicGenLanguageModel(f"facebook/musicgen-{args.model_name}", device_map=args.device)
    with nn_model.generate("Hello world!", max_new_tokens=1):
        ...
    
    distributed_state = PartialState()
    nn_model.device_map = distributed_state.device
    nn_model.to(distributed_state.device)
    with distributed_state.split_between_processes(args.ablation_layers) as layer_idxs:
        for layer_id in layer_idxs:
            path = (OUTPUT_DATA_DIR / 'ablate' / args.model_name
                    / args.dataset_name.split('/')[1]
                    / args.dataset_split / args.concept.replace(' ', '_') / str(layer_id))
            path.mkdir(exist_ok=True, parents=True)
            for batch in tqdm(batches, desc=f'l: {layer_id}'):
                prompts = batch[args.dataset_column] * args.music_per_prompt
                layer =nn_model.decoder.model.decoder.layers[layer_id]
                outputs = nn_model.generate_with_ablation(layer, prompts, args.num_tokens)
                for audio_idx in range(len(prompts)):
                    torchaudio.save(
                        path / f'out_{str(uuid4())[:6]}.wav',
                        src=outputs[0][audio_idx].detach().cpu(),
                        sample_rate=nn_model.config.sampling_rate,
                        channels_first=True,
                    )

if __name__ == "__main__":
    main()