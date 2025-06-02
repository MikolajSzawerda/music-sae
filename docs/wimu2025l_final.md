<p align="center">
  <h1 align="center">WIMU 2025L</h1>
  <h2 align="center">MusicSAE - Final Documentation</h2>
</p>

## Team
- Mikołaj Szawerda
- Mateusz Kiełbus
- Patryk Filip Gryz

## Goal

The goal of this project was to discover meaningful representations withing generative music models that enabled controlled modification of the model's output by manipualting high-level features. We focused on identyfing such representations and the presence of interpretable concepts in models like MusicGen, RAVE and YuE using XAI methods - primarly Sparse AutoEncoders (SAEs). Given the limited prior exploration of SAE for explaining synthetic music models, we also incorporated additional approaches such as activation patching [https://arxiv.org/pdf/2309.16042](https://arxiv.org/pdf/2309.16042).

## Used Technology
- python
- ruff
- uv
- just
- git
- pytorch
- accelerate :huggingface:
- datasets :huggingface:
- transformers :huggingface:

## Conducted experiments

### Rave Ablation

```
@TODO: describe in few words ablation of Rave using activation patching
```
![Rave Ablation](figures/rave_ablation.jpg)

### MusicGen Ablation
```
@TODO: describe in few words ablation for musicgen 
```
![MusicGen Ablation](figures/musicgen_ablation.png)

### YuE Ablation
```
@TODO: describe in few words ablation for yue
```
![YuE Ablation](figures/yue_ablation.png)