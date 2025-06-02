<p align="center">
  <h1 align="center">🎵 MusicSAE 🎵</h1>

  <p align="center">
    Sparse Autoencoders (SAEs) for unsupervised music representation learning.<br><br>
    <strong>🔧 Python 🧠 Deep Learning 🎧 Audio Processing 🧬 SAE 🎶 Music Embeddings</strong>
  </p>
</p>

## 📌 Description  

MusicSAE is a research-oriented project focused on exploring the potential of **Sparse Autoencoders (SAEs)** in the context of music models. In particular, it investigates how SAEs can be applied to extract meaningful laten representations from state-of-the-art generative audio models, such as **RAVE**, **MusicGen**, and **YuE**.

This project is developed as part of the **WIMU 2025L** and **ZZSN 2025L** courses at the **Warsaw University of Technology**.
Details about the schedule and scope of work can be found in the [WIMU 2025L scope](docs/wimu2025l.md) and [ZZSN 2025L scope](docs/zzsn2025l.md) files respectively.

## 👥 Team  

### Project Members

- **Mikołaj Szawerda** - Leader
- **Mateusz Kiełbus**
- **Patryk Filip Gryz**

### Contributors

Contribute to be first! 🚀  

## 📂 Project Structure  

```
/music-sae
├── conf/                     # Experiment configuration files
├── data/                     # Input datasets and processed data
├── dependencies/             # External dependencies (e.g. submodules)
├── docs/                     # Project documentation and specs
├── models/                   # Checkpoints or model definitions
├── musicsae/                 # SAE implementation for autoregressive music models
├── rave/                     # Code specific to the RAVE model integration
├── yue/                      # Code specific to the YuE model integration
├── src/                      # Shared utilities and core logic
├── notebooks/                # Research notebooks and experiments
├── scripts/                  # Scripts for training, evaluation, and automation
├── test/                     # Unit and integration tests
│
├── .env.sample               # Template for environment variables
├── .gitmodules               # Git submodules config
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
├── justfile                  # Task automation with Just
├── logging.conf              # Logging configuration
├── pyproject.toml            # Project metadata and dependencies
├── ruff.toml                 # Ruff linter configuration
├── uv.lock                   # Lock file for uv package manager
├── README.md                 # Project overview and instructions
```

## 📦 Requirements

### Mandatory
- uv
- just
- git

### Optional
- **NVIDIA drivers** — necessary if you plan to use GPU acceleration for model training




## 🚀 Usage  

1. Clone the repository with submodules
```sh
git clone --recurse-submodules -j8 git@github.com:MikolajSzawerda/music-sae.git
```
2. Install dependencies
```
just
```

For more detailed instructions and usage examples, please refer to the [User Manual](docs/user_manual.md).

## 🤝 Contribution

We welcome contributions! To maintain code quality and consistency, please make sure you have the following tools installed **before committing**:

- **pre-commit** — for automatic code formatting and checks  
- **ruff** — for linting and style enforcement  
- **just** — task runner for running tests and other automation

You can install and run pre-commit hooks with:

```sh
pip install pre-commit ruff
pre-commit install
```

Please follow the existing code style and write clear commit messages. Feel free to open issues or pull requests!

## 📜 License  
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.