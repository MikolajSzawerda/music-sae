import torchaudio
from musicsae.nnsight_model import MusicGenLanguageModel
from src.project_config import OUTPUT_DATA_DIR
from dataclasses import dataclass
from simple_parsing import list_field
from datasets import load_dataset
import logging
from uuid import uuid4
from tqdm import tqdm
from accelerate import PartialState
import hydra
from hydra.core.config_store import ConfigStore
from transformers import set_seed
logger = logging.getLogger(__name__)

@dataclass
class AblateDatasetScriptConfig:
    name: str = 'amaai-lab/MusicBench'
    split: str = 'test'
    column: str = 'main_caption'
    max_rows: int = 10

@dataclass
class AblateScriptConfig:
    dataset: AblateDatasetScriptConfig
    model_name: str = 'small'
    device: str = 'cuda'
    batch_size: int = 10
    music_per_prompt: int = 1
    ablation_layers: list[int] = list_field()
    num_tokens: int = 255
    run_with_clean: bool = False
    seed: int = 42

cs = ConfigStore.instance()
cs.store(name="config", node=AblateScriptConfig)

def get_prompt_batches(cfg: AblateScriptConfig):
    prompts_ds = load_dataset(cfg.dataset.name, split=cfg.dataset.split).select_columns([cfg.dataset.column])
    prompts_ds = prompts_ds.shuffle(cfg.seed)
    prompts_ds = prompts_ds.select(range(cfg.dataset.max_rows))
    logger.info(f"# of prompts: {prompts_ds.num_rows}")
    return prompts_ds.batch(cfg.batch_size // cfg.music_per_prompt)

@hydra.main(version_base=None, config_path="../conf/ablation", config_name="config")
def main(args: AblateScriptConfig):
    set_seed(args.seed)
    batches = get_prompt_batches(args)
    distributed_state = PartialState()
    nn_model = MusicGenLanguageModel(f"facebook/musicgen-{args.model_name}", device_map=args.device)
    with nn_model.generate("Hello world!", max_new_tokens=1):
        ...
    nn_model.device_map = distributed_state.device
    nn_model.to(distributed_state.device)
    jobs = list(args.ablation_layers) + (['pure'] if args.run_with_clean else [])
    with distributed_state.split_between_processes(jobs) as job_idxs:
        for layer_id in job_idxs:
            path = (OUTPUT_DATA_DIR / 'ablate' / args.model_name
                    / args.dataset.name.split('/')[1]
                    / args.dataset.split / str(layer_id))
            path.mkdir(exist_ok=True, parents=True)
            dataloader = tqdm(batches) if distributed_state.is_main_process and distributed_state.local_process_index == 0 else batches

            def gen_clean(prompts: list[str]):
                return nn_model.generate_clean(prompts, args.num_tokens)
            def gen_ablate(prompts: list[str]):
                layer = nn_model.decoder.model.decoder.layers[layer_id]
                return nn_model.generate_with_ablation(layer, prompts, args.num_tokens)
            gen_func = gen_clean if layer_id == 'pure' else gen_ablate

            for batch in dataloader:
                prompts = batch[args.dataset.column] * args.music_per_prompt
                outputs = gen_func(prompts)
                for audio_idx in range(len(prompts)):
                    torchaudio.save(
                        path / f'out_{str(uuid4())[:6]}.wav',
                        src=outputs[0][audio_idx].detach().cpu(),
                        sample_rate=nn_model.config.sampling_rate,
                        channels_first=True,
                    )

if __name__ == "__main__":
    main()