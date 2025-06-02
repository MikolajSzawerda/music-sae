<p align="center">
  <h1 align="center">User Manual</h1>
</p>


## Introduction

**MusicSAE** is a research project focused on exploring the use of **Sparse Autoencoders (SAE)** for musical models.  

The project focuses on three main music models:
- [RAVE](https://github.com/acids-ircam/rave)
- [MusicGen](https://github.com/facebookresearch/audiocraft)
- [YuE](https://github.com/multimodal-art-projection/YuE)

## System Requirements

### 🔧 Software Requirements
- `uv` – dependency manager ([link](https://github.com/astral-sh/uv))
- `just` – task runner ([link](https://github.com/casey/just))
- Python 3.10 or newer
- CUDA 12.x compatible drivers

### 🖥️ Hardware Requirements

- **Minimal**:  
  - GPU with at least **8 GB VRAM** (e.g. NVIDIA RTX 3060 or better)  
  - Suitable for basic experimentation and smaller models

- **Recommended**:  
  - GPU with **24 GB VRAM** (e.g. NVIDIA RTX 4090, A6000, or equivalent)  
  - Required for using MusicGen (medium) or YuE.

## Installation

```
just
```
This will:
- Set up pre-commit hooks
- Create a virtual environment using `uv`
- Download required Python version
- install dependencies from `pyproject.toml`

## Project Structure Overview

For a detailed overview of the project structure, see [README.md](./../README.md).

## Configuration

Most experiments are configured using `.yaml` files under the `conf/` directory. For model-specific options and detailed configuration guidelines, refer to the model documentation.

## Running Experiments

Use the `just` command to execute predefined experiments.

### 🔧 Example

```bash
just split-audio
```

This command splits the downloaded dataset's audio files into two stems: vocals and instrumentals using Demucs.

## Available Models

### MusicGen
```
@TODO: describe the available experiments
```

### YuE
```
@TODO: describe the available experiments 
@TODO: describe the splitting audio
```

### Rave
To use the analysis scripts for selected **RAVE** models, follow these steps:

1. **Open a terminal** and navigate to the `rave/` directory located in the root of the repository:
   [`https://github.com/MikolajSzawerda/music-sae.git`](https://github.com/MikolajSzawerda/music-sae.git)

2. Ensure you're on the `main` branch.

3. Install and activate the environment using [Poetry](https://python-poetry.org/docs/):

   ```bash
   poetry env use python3.10
   poetry install
   ```

   > 💡 If Python 3.10 is not available on your system, you may need to install it first:
   >
   > ```bash
   > sudo apt install python3.10 python3.10-venv python3.10-dev
   > ```

4. Once the environment is ready, you can run any of the RAVE analysis scripts using:

   ```bash
   poetry run python [script_name].py [arguments]
   ```

   Replace `[script_name]` and `[arguments]` with the appropriate script and options for your task.

#### Available Scripts

##### `activation_sensivity.py`

This script is used to analyze the sensitivity of a selected Rave model's layers to modifications in a chosen feature of the input data (pitch or tempo of real-valued tensors representing sound).

**Parameters**

| **Parameter** | **Description** |
| ------------- | --------------- |
| **`audio_dir`** | Path to a folder containing `.wav` files from which audio sample tensors will be extracted. |
| **`filename`** | Path to the file containing the saved Rave model. The model must be in TorchScript (\*.ts) format. |
| **`callbacks_file`** | Path to a `.py` file containing a **`getCallbacks`** function, which takes no arguments and returns a Python dictionary. Keys are **user-defined** layer names in the model, and values are dictionaries with keys **`callback`** (a Python function object to collect activations from the model layer) and **`args`** (a dictionary of arguments for that function). **Callback** functions must have at least two parameters: **`model`** (the model object) and **`batch`** (a batch of input tensors). |
| **`tested_parametr`** | The feature of input data to modify; either `pitch` or `tempo`. |
| **`params_file`** | Path to the file containing parameter values for the selected data modification function. Each line must follow the format: `[parameter_name],[parameter_value],[parameter_type]`. Example files: `pitch_params.txt` and `tempo_params.txt in the rave directory`. **`sr`** is the sample rate (Hz). |
| **`pitch_modification_steps`** | Number of semitones to shift pitch, and **`rate`** is the tempo change ratio. |
| **`audio_category`** | Music genre or user-defined category of the input data. |
| `--chunk_size` | Optional; number of audio samples (floats) in each input tensor. Default: 513.|
| `--batch_size` | Optional; number of tensors in each input batch. Default: 16\. |

**Output**
- A bar chart showing sensitivity of model layers to the selected feature. Saved in the **`diagrams`** folder.
- Data for the chart saved in the **`diagrams_data`** folder.

##### `activations.py`

This script is used to collect activations from a selected layer of a selected Rave model.

**Parameters**

| **Parameter** | **Description** |
| ------------- | --------------- |
| **`audio_dir`** | Same as in the previous script. |
| **`callbacks_file`** | Same as in the previous script. |
| **`cluster_size`** | Number of input batches processed together and saved in one output file. Multiple output files can be generated. |
| **`filename`** | Same as in the previous script. |
| **`layer_name`** | Layer name matching a key in the dictionary returned by **`getCallbacks`** from the **`callbacks_file`**. |
| **`output_name`** | Path to output files containing activation clusters, includes folder path and filename prefix (e.g., **`activations/encoder_5/encoder_5_activation_file`**). A unique ID and `.pt` extension will be appended to each file. |
| **`output_batches_name`** | Path to output files containing the corresponding input batches. Uses the same filename prefix approach as **`output_name`**. |
| **`--chunk_size`** | Same as in the previous script. |
| **`--batch_size`** | Same as in the previous script. |

**Output**
- Activation cluster files (see **`output_name`**).
- Input batch cluster files (see **`output_batches_name`**).

##### `sae_training.py`

This script trains a SAE on the collected activations using selected hyperparameters.

**Parameters**

| **Parameter** | **Description** |
| ------------- | --------------- |
| **`activations_path`** | Path to the folder containing saved activation files. |
| **`params_path`** | Path to a file with hyperparameter configurations. A sample file **`sae_params.json`** is available in the **`rave`** folder. |
| **`params_id`** | ID of the selected hyperparameter set (a key in the JSON file). |
| **`base_name`** | Common name prefix for output charts and training data, stored in the **`diagrams`** and **`diagrams_data`** folders. |
| **`output_path`** | Path to the file where the trained SAE weights will be saved. |
| **`--pretrained_weights_path`** | Optional; path to previously trained weights to resume training. |

**Output**
- SAE weight file **(`output_path`).**
- Training charts and loss data saved in the **`diagrams`** and **`diagrams_data`** folders.

##### `dictionary_learning_sae.py`

This script encodes activations using a trained SAE model. The repository also includes **`dictionary_learning.py`**, which uses the **`MiniBatchDictionaryLearning`** class from scikit-learn with similar arguments and additional ones for training epochs and validation cluster count.

**Parameters**

| **Parameter** | **Description** |
| ------------- | --------------- |
| **`activations_path`** | Same as above. |
| **`weights_path`** | Path to the trained SAE weight file. |
| **`multiply_factor`** | Factor defining the number of neurons in the SAE hidden layer. Must match the value used during training. |
| **`output_path`** | Output path for encoded activation tensors. Files contain encoded tensors without batching. |

**Output**
- Encoded activation tensors (**`output_path`**).

##### `separation_score.py`

This script evaluates how well the SAE hidden layer separates features in the encoded activations.

**Parameters**

| **Parameter** | **Description** |
| ------------- | --------------- |
| **`encoded_path`** | Path to the encoded tensor file. |
| **`batches_path`** | Path to the batch cluster file used to generate the activations that were encoded. |

**Output**
- Most frequently activated neurons per feature (pitch, tempo).
- **Silhouette coefficients** indicating feature-based clustering quality.
- Classification and regression performance for pitch and tempo using Random Forest models with cross-validation. Outputs classification accuracy and R² for regression.
- **t-SNE plots** visualizing pitch and tempo clustering.