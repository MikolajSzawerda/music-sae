# ZZSN 2025L

## Team
- Mikołaj Szawerda
- Patryk Filip Gryz

## Initial Documentation

### Description
The goal of the project is to identify common features of autoregressive generative text-to-music models.

### Approach
- SAE
- ablation study
- cross-coder to find features common to both models

### Assumptions

#### Models
In the project two models will be used:
- MusicGen ![Github](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)
- YuE ![Github](https://github.com/multimodal-art-projection/YuE)

#### Dataset
In experiments `amaai-lab/MusicBench` dataset will be used. Dataset is available on HuggingFace: ![HuggingFace](http://huggingface.co/datasets/amaai-lab/MusicBench)

#### Tools
- uv
- ruff
- just
- pytorch
- transformers 
- hydra
- accelerate

#### Expected functionality / results
We expected to find common features to both models.