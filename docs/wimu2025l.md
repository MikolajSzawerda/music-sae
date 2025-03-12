# WIMU 2025L

## Team
- Mikołaj Szawerda
- Mateusz Kiełbus
- Patryk Filip Gryz

## Design Proposal

### Description
The goal of the project is to find representations of generative music models that allow modifications to the model's output by changing high-level features, using XAI methods mainly SAEs (Sparse AutoEncoders). We will attempt to find representations of MusicGen or Rave models.

### Schedule

| Date            | Task                                                                                           |
|-----------------|------------------------------------------------------------------------------------------------|
| 20 Feb - 9 Mar  | Familiarization with materials and domain                                                      |
| 10 Mar - 16 Mar | Familiarization with bibliography/models                                                       |
| 17 Mar - 23 Mar | Getting into MusicGen / Rave models, first SAE training                                        |
| 24 Mar - 30 Mar | Training SAE on a few layers and interpreting initial results (MusicGen, Rave)                 |
| 31 Mar - 6 Apr  | Training SAE on a few layers and interpreting initial results (MusicGen, Rave) - continuation  |
| 7 Apr - 13 Apr  | Training the remaining layers and interpreting more complex concepts                           |
| 14 Apr - 20 Apr | *Easter* / Training the remaining layers and interpreting more complex concepts - continuation |
| 21 Apr - 27 Apr | *Easter* / *Reserve*                                                                           |
| 28 Apr - 4 May  | *May break* / *Reserve*                                                                        |
| 5 May - 11 May  | Attempt to disable/enable example concept                                                      |
| 12 May - 18 May | Attempt to disable/enable example concept - continuation                                       |
| 19 May - 25 May | Attempt to multiple disable/enable concepts                                                    |
| 26 May - 1 Jun  | *Polishing*                                                                                    |
| 2 Jun - 8 Jun   | *Reserve*                                                                                      |

\*Reserve - reserve week to catch up on delayed tasks

### Planned Experiments

#### Interpretability of trained SAEs  
- check if the space has semantic properties on a single layer  
- check if semantic properties are separable  
- attempt to train classifiers in the SAE space:  
  - example: provide music with and without a guitar and try to train a classifier to distinguish between these two concepts  
- attempt to identify simple concepts  
- train SAE on all layers of the music model  
- attempt to find a user-specified concept:  
  - example: the user provides recordings featuring a guitar, and the program identifies activations responsible for the presence of the guitar  

#### Modifiability of representations  
- attempt to disable a user-specified concept  
- attempt to artificially activate a user-specified concept  
- attempt to enable/disable multiple concepts simultaneously:  
  - example: the user specifies a preference for guitar music with a male vocal  

#### Utilization of different models / datasets  
- attempt to use MusicGen  
- attempt to use Rave  
- attempt to use smaller, more specific datasets and larger, more diverse datasets  

### Planned Functionalities
- concepts classification
- model output modification based on high-level concepts

### Planned Technology Stack
- python
- poetry
- ruff 
- just
- git

### Bibliography

| Title                                                                                  | Link                                                                                        | Description                                                                                                                                              |
|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| A Window Into LLMs - Sparse Autoencoders Explained                                     | https://www.youtube.com/watch?v=TKozVZoXAYs                                                 | Understanding basics of SAE, problem of polysemanticity of neurons, SAE training task                                                                    |
| Neel Nanda – Mechanistic Interpretability: A Whirlwind Tour                            | https://www.youtube.com/watch?v=veT2VI4vHyU                                                 | Mechanistic Interpretability task - decomposition into interpretable units, reverse-engineering of underlying model; neurons superpostion, SAE           |
| Demo: Gemma Scope: Sparse autoencoders on Gemma 2                                      | https://www.youtube.com/watch?v=X1gDXDQu_wU                                                 | usage of SAE in existing llm gemma architecture; SAE improvements - jumpReLU; SAE training on large scale                                                |
| Interpretability with SAEs                                                             | https://arena3-chapter1-transformer-interp.streamlit.app/[1.3.2]_Interpretability_with_SAEs | sae-lens tutorials; understanding sae circuits - information flow through different transformer layers; Feature Splitting - features appears in clusters |
| SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders | https://arxiv.org/pdf/2501.18052                                                            | Usage of SAE to modify output of generative model;                                                                                                       |
| Sparse Autoencoder Insights on Voice Embeddings                                        | https://arxiv.org/pdf/2502.00127                                                            | SAE on non-transformer architecutre; SAE exhibit characteristics similar like in LLM's; most audio features are on spectrum rather than discrete         |
| Unpacking SDXL Turbo: Interpreting Text-to-Image Models with Sparse Autoencoders       | https://arxiv.org/pdf/2410.22366                                                            | usage of SAE in image generative model; technique of visualization features causal effects                                                               |
| Are Sparse Autoencoders Useful? A Case Study in Sparse Probing                         | https://arxiv.org/pdf/2502.16681                                                            | Limitation of SAE's, efficiency of SAE depends on strongly on data and target model                                                                      |

@TODO: Lorem ipsum
