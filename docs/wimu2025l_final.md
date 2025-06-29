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
For the RAVE model, due to its architecture, a naive ablation study was not feasible, so we applied activation patching. We analyzed the effect of pitch shifts and found that decoder layers 5 and 7 had the strongest impact.
```
![Rave Ablation](figures/rave_ablation.jpg)

### MusicGen Ablation
```
An ablation study on MusicGen shows that layers 13 and 16 significantly impact output quality. Layer 16 likely captures high-level features, while early layers focus on input decoding.
```
![MusicGen Ablation](figures/musicgen_ablation.png)


### YuE Ablation
```
In YuE, a similar pattern to MusicGen is observed, with layer 15 capturing a wide range of features. This suggests it encodes high-level representations critical to generation quality.
```
![YuE Ablation](figures/yue_ablation.png)

### Collecting activations

As proofed in the experiments there was requirment to use diverse audio/prompt datasets. We have choosen four popular datasets:
- fma
- mtg-jamendo
- music-bench
- song-describer

Due to YuE’s architecture, which processes vocal and instrumental tokens separately, we split the audio into vocal and instrumental stems. This also enabled the use of instrumental tracks in MusicGen instead of full audio, improving SAE variance by 0.1.

To support other datasets in the future we introduced *plugin architecture* which allows to easily write custom transformations and loading for given dataset and use it with already present ones.

Additionaly we have used *accelerate* library to easily support mutli-gpu execution.

We used the nnsight library to trace the models without generating new tokens, which significantly speed up activation collection. This approach also allowed us to access, save, and intervene on activations directly, additionally enabling efficient ablation studies.

We used dictionary_learning as the main implementation of SAE, as it provides all the necessary functionalities required for our project, including feature extraction.

YuE operates on interleaved vocal and instrumental tokens; since the other two models generate only instrumental music, we focused our analysis on the instrumental part. To achieve this, we extracted activations at positions corresponding to instrumental tokens.


### Training SAE

We have trained several SAE models for musicgen-small and medium.

**MusicGen-Medium**

During most of our trainings we were traing to increase results of musicgen-medium model as even after adding more diverse dataset variance was never exceeding certain value.
![alt text](figures/medium_fail.png)

After lots of experiments it turned out that our activations normalization phase was to short which resulted in poor training results. After increasing normalization steps + using only instrumentals + finding best *k vs expansion factor* + enough and diverse data we managed to achieve > 0.9 explained variance.

![alt text](figures/training_medium.png)

![alt text](figures/parralel_medium.png)

**MusicGen-Small**

![alt text](figures/small_training.png)

**Yue**


During SAE training for the YuE model, we also observed that a lower expansion factor (EF) increases the SAE variance. Enabling activation normalization slightly increases the SAE variance.

![alt text](figures/yue_training.png)

### SAE interpretation

After succesfull SAE training we performed automatic feature labeling process:
- for each track activations we aggregated mean activation with respect to time
- we calculated corpus-level feature activation rate (when `mean activation > 0`)
- we filtered features which activation rate didn't meet `0.01 < act_rate < 0.25`
- for each feature we have grouped top 10 tracks with respect to their activation mean

After manual listening of some features we have noticed that tracks despite being from completely different recordings had simillar features.

We have used multimodal LLMs to automatically find common features for those subsets. We used Qwen and gemini-2.0-flash. After labeling, we calculated the mean CLAP score between text descriptions and audio embeddings, and for the best ones, we manually listened to them.

![](figures/sae_interp.png)

| Feature Name | Description |
|--------------|-------------|
| f5576 | Uptempo electronic dance music. |
| f2933 | The snippets all feature the prominent use of bells or bell-like sounds, creating a resonant and percussive soundscape. |
| f4666 | A collection of clips share the characteristic of featuring string instruments playing in a major key at a moderate tempo. |
| f2587 | A collection of short clips featuring prominently string instruments playing melodic lines with acoustic timbres. |
| f4788 | The audio clips share a common musical element centered around acoustic string instruments, particularly plucked instruments like guitar and mandolin, with a folk music influence. |
| f1393 | The audio snippets feature percussive instruments and demonstrate an energetic rhythmic style with syncopation. |
| f4798 | Collection of songs exhibiting traditional Middle Eastern melodic and rhythmic characteristics, prominently featuring wind instruments and danceable rhythms. |
| f2255 | The clips feature brass instruments in what sounds like classical or light orchestral arrangements. |
| f5235 | A collection of reggae-influenced music characterized by slow tempos and prominent bass lines. |
| f4606 | The audio clips feature elements common in Balkan music. Characteristics are complex rhythms, wind and string instruments, and fast, danceable tempos. |


### SAE interventions

We experimented with interventions on small and medium models using the following algorithms:
- Activating a specific SAE feature before every nth token and decoding the result.
- Modulating SAE features triggered by a given prompt by multiplying them with negative, zero, or positive values and observing the effect.
- Transferring activation patterns of feature A from one track to a generated track prompted with feature B.

Unfortunately, the output quality was not as crisps as we expected, indicating a need to improve dataset diversity and quality.

## Limitations and discovered issues

During the project, we encountered several challenges and limitations:
- Audio datasets often do not provide separate vocal and instrumental stems, requiring audio separation (e.g., using Demucs) which adds a time-consuming preprocessing step before activations collection.
- SAEs require a large number of activations for training, which must be generated in advance due to the slow generation process. They also require a large number of activations, necessitating the use of multiple datasets.
- The size of collected activations, even for small datasets, is substantial and can reach hundreds of gigabytes.
- Training SAE takes several hours even with a relatively small amount of activationss (e.g., 500 GB) and demands a high-speed storage device for efficient data access.
- Searching for features in SAEs demands a large amount of RAM.