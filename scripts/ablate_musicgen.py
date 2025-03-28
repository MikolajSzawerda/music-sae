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
#
# clear_output()
# # -
#
# tokens = 255
# prompt = "Recreate the essence of a classic video game theme with chiptune sounds and nostalgic melodies."
# for n in range(0, 20, 2):
#     ablate_layer = nn_model.decoder.model.decoder.layers[n]
#     with nn_model.generate(prompt, max_new_tokens=tokens):
#         outputs = nnsight.list().save() # Initialize & .save() nnsight list
#         for _ in range(tokens):
#             ablate_layer.output[0][:] = ablate_layer.input[0][:]
#             outputs.append(nn_model.generator.output)
#             nn_model.next()
#     torchaudio.save(
#         f"out_{n}.wav",
#         src=outputs[0][0].detach().cpu(),
#         sample_rate=nn_model.config.sampling_rate,
#         channels_first=True,
#     )

@dataclass
class AblateScriptConfig(Serializable):
    model_name: str = 'small'
    device: str = 'cuda'

def main():
    args = parse(AblateScriptConfig)
    nn_model = MusicGenLanguageModel(f"facebook/musicgen-{args.model_name}", device_map=args.device)

    # nn_model = MusicGenLanguageModel("facebook/musicgen-small", config=cfg, tokenizer=processor.tokenizer, device_map='cuda')
    with nn_model.generate("Hello world!", max_new_tokens=1):
        ...

if __name__ == "__main__":
    main()