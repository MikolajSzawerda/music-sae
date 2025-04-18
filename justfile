default: prepare-env

prepare-env:
    uvx pre-commit install
    uv sync

musicgen-ablation-generation:
    uv run accelerate launch scripts/ablate_musicgen.py

musicgen-ablation-fad generations_dir score_path:
    #!/bin/sh
    cd dependencies/fadtk
    for item in $(seq 1 24); do 
        uv run fadtk --inf clap-laion-audio fma_pop {{ generations_dir }}/$item {{ score_path }};
    done
    uv run fadtk --inf clap-laion-audio fma_pop {{ generations_dir }}/pure {{ score_path }};

musicgen-ablation-relative-fad generations_dir score_path:
    #!/bin/sh
    cd dependencies/fadtk
    for item in $(seq 1 24); do 
        uv run fadtk --inf clap-laion-audio {{ generations_dir }}/pure {{ generations_dir }}/$item {{ score_path }};
    done

prepare-music-bench:
	wget -O data/raw/MusicBench.tar.gz https://huggingface.co/datasets/amaai-lab/MusicBench/resolve/main/MusicBench.tar.gz
	mkdir data/input/music-bench
	tar -xzf data/raw/MusicBench.tar.gz -C data/input/music-bench/
