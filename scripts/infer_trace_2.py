from dataclasses import dataclass

import hydra
from nnsight import LanguageModel
import torch
from transformers import AutoModelForCausalLM, LogitsProcessorList

from yue.common import initialize_seed, split_lyrics, BlockTokenRangeProcessor
from yue.yue import YuEInferenceConfig, YuEProcessorConfig, YuEProcessor


@dataclass
class CollectScriptConfig:
    device: str = "cuda"
    model: str = "7B-anneal-en-cot"
    layer: int = 13
    min_new_tokens: int = 3000
    max_new_tokens: int = 3000
    seed: int = 42
    inference: YuEInferenceConfig = YuEInferenceConfig()
    processor: YuEProcessorConfig = YuEProcessorConfig()


@hydra.main(version_base=None, config_path="../conf/yue-sae/", config_name="config")
def main(args: CollectScriptConfig):
    initialize_seed(args.seed)
    device = torch.device(args.device)

    processor = YuEProcessor(device, args.processor)

    model = AutoModelForCausalLM.from_pretrained(
        f"m-a-p/YuE-s1-{args.model}", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model = LanguageModel(model, input_names=["input_ids"])
    model.to(device)
    model.eval()

    # region INPUT @TODO: replace with loading from datasets
    def load_audio_mono(filepath, sampling_rate=16000):
        import torchaudio
        from torchaudio.transforms import Resample

        audio, sr = torchaudio.load(filepath)
        # Convert to mono
        audio = torch.mean(audio, dim=0, keepdim=True)
        # Resample if needed
        if sr != sampling_rate:
            resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
            audio = resampler(audio)
        return audio

    MUSIC_PATH = "placeholder.mp3"

    audio_prompt = load_audio_mono(MUSIC_PATH)
    genres = """new wave pandeira relaxed human happy vocal""".strip()
    lyrics = split_lyrics("""[instrumental]

[instrumental]

[verse]
Staring at the sunset, colors paint the sky
Thoughts of you keep swirling, can't deny
I know I let you down, I made mistakes
But I'm here to mend the heart I didn't break

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light
You can't fight this feeling now
I won't back down
You know you can't deny it now
I won't back down

[verse]
They might say I'm foolish, chasing after you
But they don't feel this love the way we do
My heart beats only for you, can't you see?
I won't let you slip away from me

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light
You can't fight this feeling now
I won't back down
You know you can't deny it now
I won't back down

[bridge]
No, I won't back down, won't turn around
Until you're back where you belong
I'll cross the oceans wide, stand by your side
Together we are strong

[outro]
Every road you take, I'll be one step behind
Every dream you chase, love's the tie that binds
You can't fight this feeling now
I won't back down""")
    # endregion

    # region CHOOSE LAYER
    layer = model.model.layers[args.layer]

    with torch.no_grad():
        inputs = processor.process(genres, lyrics, audio_prompt)

        with model.trace(
            inputs=inputs,
            max_new_tokens=args.max_new_tokens,
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
            trace = layer.output[0].save()

        print(trace.shape)
        # return trace
        # @TODO: in this place take the trace and save activations

        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
