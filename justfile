default: prepare-env

prepare-env:
    poetry install

musicgen-ablation-generation:
    poetry run accelerate launch scripts/ablate_musicgen.py