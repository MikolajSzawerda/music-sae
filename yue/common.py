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

import os
from contextlib import contextmanager
import json
import random
import re

import numpy as np
import torch
from transformers import LogitsProcessor


def initialize_seed(seed: int = 42) -> None:
    """
    Initializes the random seed for reproductibility across common libraries.

    Sets the seed for `random`, NumPy and PyTorch.

    Parameters
    ----------
    seed : int = 42
        The seed value to sued

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores


def split_lyrics(lyrics: str) -> list[str]:
    """
    Splits the lyrics

    Returns
    -------
    list[str]
        List of lyrics
    """
    pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics


@contextmanager
def temporary_cwd(path):
    previous_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def load_tags() -> list[str]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tags.json")

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    return data


def filter_tags(tags: list[str], text: str) -> str:
    text = text.lower()
    filtered = [tag for tag in tags if tag in text]
    return " ".join(filtered)


def get_instrumental_only_lyrics() -> str:
    instrumental_only = "[verse]\n\n\n\n\n \n[chorus]\n\n\n\n\n[chorus]\n\n\n\n\n[outro]"
    return split_lyrics(instrumental_only)
