"""
Training dictionaries
"""

import json
import torch.multiprocessing as mp
import os
from typing import Optional
from contextlib import nullcontext

import torch as t
from tqdm import tqdm
from random import choice

from dictionary_learning.training import new_wandb_process


def log_stats(
    trainers,
    step: int,
    act: dict[str, t.Tensor],
    activations_split_by_head: bool,
    transcoder: bool,
    model_name: str,
    log_queues: list = [],
    verbose: bool = False,
):
    with t.no_grad():
        # quick hack to make sure all trainers get the same x
        for i, trainer in enumerate(trainers):
            log = {}
            x_act, act_hat, f, losslog = trainer.loss(act, model_name, step=step, logging=True)

            # L0
            l0 = (f != 0).float().sum(dim=-1).mean().item()
            # fraction of variance explained
            total_variance = t.var(x_act, dim=0).sum()
            residual_variance = t.var(x_act - act_hat, dim=0).sum()
            frac_variance_explained = 1 - residual_variance / total_variance
            log[f"frac_variance_explained_{model_name}"] = frac_variance_explained.item()

            if verbose:
                print(f"[{model_name}] Step {step}: L0 = {l0}, frac_variance_explained = {frac_variance_explained}")

            # log parameters from training
            log.update(
                {f"{model_name}_{k}": v.cpu().item() if isinstance(v, t.Tensor) else v for k, v in losslog.items()}
            )
            log[f"{model_name}_l0"] = l0
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                if isinstance(value, t.Tensor):
                    value = value.cpu().item()
                log[f"{model_name}_{name}"] = value

            if log_queues:
                log_queues[i].put(log)


def get_norm_factor(data, steps: int) -> dict[str, float]:
    total_mean_squared_norm = {}
    count = 0

    for step, act_BD in enumerate(tqdm(data, total=steps, desc="Calculating norm factor")):
        if step > steps:
            break
        count += 1
        for name, act in act_BD.items():
            mean_squared_norm = t.mean(t.sum(act**2, dim=1))
            total_mean_squared_norm[name] = total_mean_squared_norm.get(name, 0) + mean_squared_norm

    average_mean_squared_norm = {k: v / count for k, v in total_mean_squared_norm.items()}
    norm_factor = {k: t.sqrt(v).item() for k, v in average_mean_squared_norm.items()}

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factor: {norm_factor}")

    return norm_factor


def trainSAE(
    data,
    trainer_configs: list[dict],
    steps: int,
    use_wandb: bool = False,
    wandb_entity: str = "",
    wandb_project: str = "",
    save_steps: Optional[list[int]] = None,
    save_dir: Optional[str] = None,
    log_steps: Optional[int] = None,
    activations_split_by_head: bool = False,
    transcoder: bool = False,
    run_cfg: dict = {},
    normalize_activations: bool = False,
    verbose: bool = False,
    device: str = "cuda",
    autocast_dtype: t.dtype = t.float32,
):
    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = (
        nullcontext() if device_type == "cpu" else t.autocast(device_type=device_type, dtype=autocast_dtype)
    )

    trainers = []
    for i, config in enumerate(trainer_configs):
        if "wandb_name" in config:
            config["wandb_name"] = f"{config['wandb_name']}_trainer_{i}"
        trainer_class = config["trainer"]
        del config["trainer"]
        trainers.append(trainer_class(**config))

    wandb_processes = []
    log_queues = []

    if use_wandb:
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue()
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            # Make sure wandb config doesn't contain any CUDA tensors
            wandb_config = {k: v.cpu().item() if isinstance(v, t.Tensor) else v for k, v in wandb_config.items()}
            wandb_process = mp.Process(
                target=new_wandb_process,
                args=(wandb_config, log_queue, wandb_entity, wandb_project),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)

    # make save dirs, export config
    if save_dir is not None:
        save_dirs = [os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            # save config
            config = {"trainer": trainer.config}
            with open(os.path.join(dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    if normalize_activations:
        norm_factor = get_norm_factor(data, steps=100)

        for trainer in trainers:
            trainer.config["norm_factor"] = norm_factor
            # Verify that all autoencoders have a scale_biases method
            trainer.ae.scale_biases(1.0)

    for step, act in enumerate(tqdm(data, total=steps)):
        act = {k: v.to(dtype=autocast_dtype, device=device) for k, v in act.items()}

        if normalize_activations:
            act = {k: v / norm_factor[k] for k, v in act.items()}

        if step >= steps:
            break

        # logging
        if (use_wandb or verbose) and step % log_steps == 0:
            for name in act.keys():
                log_stats(
                    trainers,
                    step,
                    act,
                    activations_split_by_head,
                    transcoder,
                    name,
                    log_queues=log_queues,
                    verbose=verbose,
                )

        # saving
        if save_steps is not None and step in save_steps:
            for dir, trainer in zip(save_dirs, trainers):
                if dir is not None:
                    if normalize_activations:
                        # Temporarily scale up biases for checkpoint saving
                        trainer.ae.scale_biases_by_name(norm_factor)

                    if not os.path.exists(os.path.join(dir, "checkpoints")):
                        os.mkdir(os.path.join(dir, "checkpoints"))

                    checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                    t.save(
                        checkpoint,
                        os.path.join(dir, "checkpoints", f"ae_{step}.pt"),
                    )

                    if normalize_activations:
                        trainer.ae.scale_biases_by_name({k: 1 / v for k, v in norm_factor.items()})

        # training
        model_name = choice(list(act.keys()))
        for trainer in trainers:
            with autocast_context:
                trainer.update(step, act, model_name)

    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if normalize_activations:
            trainer.ae.scale_biases_by_name(norm_factor)
        if save_dir is not None:
            final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
            t.save(final, os.path.join(save_dir, "ae.pt"))

    # Signal wandb processes to finish
    if use_wandb:
        for queue in log_queues:
            queue.put("DONE")
        for process in wandb_processes:
            process.join()
