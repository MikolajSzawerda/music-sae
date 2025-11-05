# SIWY 2025Z

## Team
- Mikołaj Szawerda
- Patryk Filip Gryz
- Anna Schäfer

## Design Proposal

### Description
The goal of the project is to extend the previous research with the quality evaluation of trained model, visualization of the model output and research, usage and comparison of alternative methods to our previous method.

### Schedule

| Date            | Task | 
| --------------- | ---- | 
|  3 Nov -  9 Nov | ?    |
| 10 Nov - 16 Nov | ?    |
| 17 Nov - 23 Nov | ?    |
| 24 Nov - 30 Nov | ?    |
| 24 Nov - 30 Nov | ?    |
|  1 Dec -  7 Dec | ?    |
|  8 Dec - 14 Dec | ?    |
| 15 Dec - 21 Dec | ?    |
| 22 Dec - 28 Dec | ?    |
| 29 Dec -  4 Jan | ?    |
|  5 Jan - 11 Jan | ?    |
| 12 Jan - 18 Jan | ?    |
| 19 Jan - 23 Jan | *Reserve* |

\*Reserve - reserve week to catch up on delayed tasks

### Planned Experiments

#### Evaluation of meaningfulness of musicSAE outputs
- check if output of the model has sens
- check if meaningfulness of the trained model is better than simpler methods
- check how data influence the meaningfulness of the model outputs

#### Second
??

### Planned Functionalities
- implementation of method(s) to evaluate the quality or meaningfulness of model outputs - accessing whether the explanations make sens
    - baseline approach using sparse autoencoders (SAE).

### Planned Technology Stack
**General**
- python
- uv
- ruff
- just
- git
- hydra

**NN-specific**
- nnsight
- wandb
- huggingface (datasets, transformers)
- torch


### Bibliography

| Title | Link | Description |
| ----- | ---- | ----------- | 
| SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders | https://arxiv.org/pdf/2501.18052                                                            | Usage of SAE to modify output of generative model;                                                                                                       |
| Are Sparse Autoencoders Useful? A Case Study in Sparse Probing                         | https://arxiv.org/pdf/2502.16681                                                            | Limitation of SAE's, efficiency of SAE depends on strongly on data and target model|
| | https://arxiv.org/abs/2405.08366 | |
| | https://openreview.net/forum?id=HpUs2EXjOl | |
| | https://arxiv.org/abs/2509.23717v1 | |