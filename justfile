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

collect-musicgen-activations:
    #!/bin/sh
    uv run python scripts/collect_musicgen_activations.py

train-musicgen-sae:
    #!/bin/sh
    uv run python scripts/train_sae_disk.py

prepare-music-bench:
	wget -O data/raw/MusicBench.tar.gz https://huggingface.co/datasets/amaai-lab/MusicBench/resolve/main/MusicBench.tar.gz
	mkdir data/input/music-bench
	tar -xzf data/raw/MusicBench.tar.gz -C data/input/music-bench/

split-music-bench:
    uv run accelerate launch --gpu_ids="0,1" --main_process_port=29501 scripts/split_audio.py +audio_input_dir=data/input/music-bench/datashare +vocals_output_dir=data/input/music-bench/datashare-vocals +instruments_output_dir=data/input/music-bench/datashare-instruments +verify=True +verify_workers=16

split-song-describer:
    uv run accelerate launch --gpu_ids="0,1" --main_process_port=29502 scripts/split_audio.py +audio_input_dir=data/input/song-describer/datashare +vocals_output_dir=data/input/song-describer/datashare-vocals +instruments_output_dir=data/input/song-describer/datashare-instruments +verify=True +verify_workers=16

split-mtg-jamendo:
    uv run accelerate launch --gpu_ids="0,1" --main_process_port=29503 scripts/split_audio.py +audio_input_dir=data/input/mtg-jamendo/datashare +vocals_output_dir=data/input/mtg-jamendo/datashare-vocals +instruments_output_dir=data/input/mtg-jamendo/datashare-instruments +verify=True +verify_workers=16
run-intervention-eval:
    uv run python3 scripts/eval_interventions.py data/output/sae_interventions/run_20250806_211548_348865cd/ --output data/output/sae_interventions/results.json
run-intervention-gen:
    uv run python3 scripts/generate_interventions.py --features 4606 5235 2255 4798 1393 4788 2587 4666 2933 5576 --max_tokens 200 --intervention_frequency 2 --intervention_value "-2" --algorithm set_value --generate_baseline_clean --generate_baseline_passthrough
run-filtered-act:
    CUDA_VISIBLE_DEVICES=1 uv run python3 scripts/collect_musicgen_activations.py --config-name="config-jamendo-anty-reggae"