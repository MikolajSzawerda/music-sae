# ZZSN 2025L

## Team
- Mikołaj Szawerda
- Patryk Filip Gryz

## Initial Documentation

### Description
The goal of the project is to identify common features of autoregressive generative text-to-music models.

### Approach
- ablation study of layers
- SAE - TopK version
- cross-coder - USAE - to find features common to both models

We will first perform ablation study to select layers that activations will be worth of collecting. We will then gather activations from models using datasets(best in training mode inputing audio+prompt instead of generation).
As the next step we will find best sae traing parameters to make sure final model will have enough capacity for common space. Having trained sae we will construct USAE with two encoders and decoders and train as in ![paper](https://arxiv.org/abs/2502.03714) by encoding activation from random model and calculate loss on all decoders.

Having trained USAE we will use Activation Maximization technique to find common and divergent features + we will compare features learned by indepentend SAE and USAE.

In best case scenerio we want to compare Yue and MusicGen-medium. However, because Yue is novel model and additionaly its implementation can be characterized as "research code" we will compare as a baseline musicgen-small and medium.

### Assumptions

#### Models
In the project two models will be used:
- MusicGen ![Github](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md) (most probably in two versions: small; medium)
- YuE ![Github](https://github.com/multimodal-art-projection/YuE)

#### Dataset
In experiments `amaai-lab/MusicBench` dataset will be used. Dataset is available on HuggingFace: ![HuggingFace](http://huggingface.co/datasets/amaai-lab/MusicBench). Alternativly for more prompts also ![Song Describer](https://huggingface.co/datasets/renumics/song-describer-dataset) and ![MTG-Jamendo](https://mtg.github.io/mtg-jamendo-dataset/).

#### Tools
- uv
- ruff
- just
- pytorch
- transformers 
- hydra
- accelerate
- dictionary_learning
- nnsight
- wandb

#### Expected functionality / results
We expected to find common features to both models.
