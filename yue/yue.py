"""
Source: https://github.com/multimodal-art-projection/YuE
This file includes source code derived from YuE
The original code is licensed under the Apache License, Version 2.0

---

Copyright 2025 HKUST and M-A-P

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# region Load local models
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dependencies/xcodec_mini_infer"))
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dependencies/xcodec_mini_infer", "descriptaudiocodec")
)
# endregion

from dataclasses import dataclass

from models.soundstream_hubert_new import SoundStream
import numpy as np
from omegaconf import OmegaConf
import torch
from transformers import BatchEncoding

from .common import temporary_cwd
from .codecmanipulator import CodecManipulator
from .mmtokenizer import MMSentencePieceTokenizer


@dataclass
class YuEInferenceConfig:
    top_p: float = 0.93
    temperature: float = 1.0
    repetition_penalty: float = 1.1
    guidance_scale: float = 1.5


@dataclass
class YuEProcessorConfig:
    tokenizer_model: str = "./mm_tokenizer_v0.2_hf/tokenizer.model"
    codec_parent_path: str = "./"
    codec_config: str = "xcodec_mini_infer/final_ckpt/config.yaml"
    codec_resume: str = "xcodec_mini_infer/final_ckpt/ckpt_00360000.pth"


class YuEProcessor:
    def __init__(self, device, config: YuEProcessorConfig = None):
        if config is None:
            config = YuEProcessorConfig()

        codec_model_config = OmegaConf.load(os.path.join(config.codec_parent_path, config.codec_config))
        codec_parameter_dict = torch.load(
            os.path.join(config.codec_parent_path, config.codec_resume), map_location="cpu", weights_only=False
        )

        self._device = device
        self._tokenizer = MMSentencePieceTokenizer(config.tokenizer_model)
        self._codectool = CodecManipulator("xcodec", 0, 1)

        with temporary_cwd(config.codec_parent_path):
            self._codec_model = SoundStream(**codec_model_config.generator.config).to(self._device)

        self._codec_model.load_state_dict(codec_parameter_dict["codec_model"])
        self._codec_model.to(device)
        self._codec_model.eval()

        self._sos = self._tokenizer.tokenize("[start_of_segment]")

    @property
    def eoa(self):
        return self._tokenizer.eoa

    def _encode_audio(self, audio, target_bw=0.5):
        if len(audio.shape) < 3:
            audio.unsqueeze_(0)

        with torch.no_grad():
            raw_codes = self._codec_model.encode(audio.to(self._device), target_bw=target_bw)

        raw_codes = raw_codes.transpose(0, 1)
        raw_codes = raw_codes.cpu().numpy().astype(np.int16)
        return raw_codes

    def process(self, genres: str, lyrics: list[str], audio):
        full_lyrics = "\n".join(lyrics)
        segment = lyrics[0]
        prompt = f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"

        raw_codes = self._encode_audio(audio, target_bw=0.5)
        code_ids = self._codectool.npy2ids(raw_codes[0])
        print(len(code_ids))
        audio_prompt = [self._tokenizer.soa] + self._codectool.sep_ids + code_ids + [self._tokenizer.eoa]

        sentence_ids = (
            self._tokenizer.tokenize("[start_of_reference]")
            + audio_prompt
            + self._tokenizer.tokenize("[end_of_reference]")
        )
        head_id = self._tokenizer.tokenize(prompt) + sentence_ids

        prompt_ids = (
            head_id + self._sos + self._tokenizer.tokenize(segment) + [self._tokenizer.soa] + self._codectool.sep_ids
        )
        input_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(self._device)

        attention_mask = (input_ids != 0).long()
        inputs = BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})

        return inputs
