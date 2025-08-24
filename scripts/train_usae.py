"""
Training Universal SAEs (USAEs)
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
    trainer,
    step: int,
    act: dict[str, t.Tensor],
    log_queue: Optional[mp.Queue] = None,
    verbose: bool = False,
):
    """Log statistics for USAE training."""
    with t.no_grad():
        log = {}

        # Log stats for each model in the batch
        for model_name in act.keys():
            loss_log = trainer.loss(act, model_name, step=step, logging=True)
            x_act, act_hat, f = loss_log.x, loss_log.x_hat, loss_log.f

            # L0
            l0 = (f != 0).float().sum(dim=-1).mean().item()

            # fraction of variance explained for this model
            def frac_var(name: str):
                total_variance = t.var(x_act[name], dim=0).sum()
                residual_variance = t.var(x_act[name] - act_hat[name], dim=0).sum()
                return 1 - residual_variance / total_variance

            frac_variance_explained = frac_var(model_name)
            log[f"frac_variance_explained_{model_name}"] = frac_variance_explained.item()
            for name in x_act.keys():
                if name != model_name:
                    log[f"frac_variance_explained_{model_name}->{name}"] = frac_var(name).item()
            log[f"l0_{model_name}"] = l0

            # Add model-specific loss components
            for loss_key, loss_value in loss_log.losses.items():
                log[f"{model_name}_{loss_key}"] = loss_value

            if verbose:
                print(
                    f"[{model_name}] Step {step}: L0 = {l0:.2f}, frac_variance_explained = {frac_variance_explained:.4f}"
                )

        # Add trainer-specific logging parameters (shared across models)
        trainer_log = trainer.get_logging_parameters()
        for name, value in trainer_log.items():
            if isinstance(value, t.Tensor):
                value = value.cpu().item()
            log[f"trainer_{name}"] = value

        if log_queue:
            log_queue.put(log)


def get_norm_factor(data, steps: int) -> dict[str, float]:
    """Calculate normalization factors for each model."""
    total_mean_squared_norm = {}
    count = 0

    for step, act_BD in enumerate(tqdm(data, total=steps, desc="Calculating norm factor")):
        if step >= steps:
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


def save_checkpoint(trainer, save_dir: str, step: int, norm_factor: dict = None, is_final: bool = False):
    """Save a training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    # Temporarily scale biases for saving if needed
    if norm_factor:
        trainer.ae.scale_biases_by_name(norm_factor)

    # Save complete trainer state
    if hasattr(trainer, "save_trainer_state"):
        # USAE trainer - save complete state
        checkpoint_data = trainer.save_trainer_state()
        checkpoint_data.update({"step": step, "norm_factor": norm_factor, "is_final": is_final})

        # Move tensors to CPU for saving
        def move_to_cpu(obj):
            if isinstance(obj, dict):
                return {k: move_to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, t.Tensor):
                return obj.cpu()
            else:
                return obj

        checkpoint_data = move_to_cpu(checkpoint_data)
    else:
        # Standard trainer - save model state only
        checkpoint_data = {
            "ae_state_dict": {k: v.cpu() for k, v in trainer.ae.state_dict().items()},
            "trainer_config": trainer.config,
            "step": step,
            "norm_factor": norm_factor,
            "is_final": is_final,
        }

    if is_final:
        save_path = os.path.join(save_dir, "ae.pt")
    else:
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, f"ae_{step}.pt")

    t.save(checkpoint_data, save_path)

    # Scale biases back if needed
    if norm_factor:
        trainer.ae.scale_biases_by_name({k: 1 / v for k, v in norm_factor.items()})

    print(f"Saved {'final model' if is_final else 'checkpoint'} to {save_path}")


def load_checkpoint(checkpoint_path: str, trainer_class, device: str = "cuda"):
    """Load a training checkpoint and return trainer and metadata."""
    print(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint data
    checkpoint_data = t.load(checkpoint_path, map_location=device)

    # Check if this is new format (complete trainer state) or old format
    if "trainer_states" in checkpoint_data and hasattr(trainer_class, "load_trainer_state"):
        # New format with complete trainer state
        print("Loading USAE trainer with complete state...")
        trainer = trainer_class.load_trainer_state(checkpoint_data, device)

        metadata = {
            "step": checkpoint_data.get("step", 0),
            "norm_factor": checkpoint_data.get("norm_factor"),
            "is_final": checkpoint_data.get("is_final", False),
        }
    else:
        # Old format or fallback - try to reconstruct
        print("Loading with fallback method...")

        # Try to extract config from checkpoint or look for old metadata file
        trainer_config = checkpoint_data.get("trainer_config")
        if not trainer_config:
            metadata_path = checkpoint_path.replace(".pt", "_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                trainer_config = metadata.get("trainer_config", {})
            else:
                raise ValueError("No trainer config found. Cannot resume training.")

        if not trainer_config:
            raise ValueError("No trainer config found in checkpoint. Cannot resume training.")

        # Create trainer with saved config
        init_config = {
            k: v for k, v in trainer_config.items() if k not in ["trainer_class", "dict_class", "activation_dim"]
        }
        init_config["device"] = device

        # Handle legacy configs that might have activation_dim instead of activation_dims
        if hasattr(trainer_class, "load_trainer_state") and "activation_dims" not in init_config:
            # This is a USAE trainer but config is missing activation_dims
            print("Warning: Reconstructing activation_dims from model state...")
            ae_state_dict = checkpoint_data.get("ae_state_dict", checkpoint_data)
            activation_dims = {}
            for key in ae_state_dict.keys():
                if key.startswith("saes.") and key.endswith(".decoder.weight"):
                    model_name = key.split(".")[1]
                    if model_name not in activation_dims:
                        activation_dims[model_name] = ae_state_dict[key].shape[0]
            init_config["activation_dims"] = activation_dims

        trainer = trainer_class(**init_config)

        # Load model state
        ae_state_dict = checkpoint_data.get("ae_state_dict", checkpoint_data)
        trainer.ae.load_state_dict(ae_state_dict)
        trainer.ae.to(device)

        metadata = {
            "step": checkpoint_data.get("step", 0),
            "norm_factor": checkpoint_data.get("norm_factor"),
            "is_final": checkpoint_data.get("is_final", False),
        }

    # Scale biases if norm factor was used
    norm_factor = metadata.get("norm_factor")
    if norm_factor:
        trainer.ae.scale_biases_by_name({k: 1 / v for k, v in norm_factor.items()})

    start_step = metadata.get("step", 0) + 1

    print(f"Loaded checkpoint from step {start_step - 1}")
    return trainer, metadata, start_step


def trainUSAE(
    data,
    trainer_config: dict,  # Single trainer config, not list
    steps: int,
    use_wandb: bool = False,
    wandb_entity: str = "",
    wandb_project: str = "",
    save_steps: Optional[list[int]] = None,
    save_dir: Optional[str] = None,
    log_steps: Optional[int] = None,
    run_cfg: dict = {},
    normalize_activations: bool = False,
    verbose: bool = False,
    device: str = "cuda",
    autocast_dtype: t.dtype = t.float32,
    resume_from: Optional[str] = None,  # Path to checkpoint to resume from
):
    """
    Train Universal SAE using a single trainer.

    Args:
        data: Iterator yielding dicts of {model_name: activations}
        trainer_config: Single trainer configuration dict
        resume_from: Optional path to checkpoint to resume training from
    """
    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = (
        nullcontext() if device_type == "cpu" else t.autocast(device_type=device_type, dtype=autocast_dtype)
    )

    # Initialize or resume trainer
    start_step = 0
    metadata = {}

    if resume_from:
        trainer_class = trainer_config["trainer"]
        trainer, metadata, start_step = load_checkpoint(resume_from, trainer_class, device)
        print(f"Resuming training from step {start_step}")
    else:
        # Create new trainer
        trainer_class = trainer_config["trainer"]
        config = trainer_config.copy()
        del config["trainer"]
        trainer = trainer_class(**config)

    # Setup wandb
    wandb_process = None
    log_queue = None

    if use_wandb:
        log_queue = mp.Queue()
        wandb_config = trainer.config | run_cfg
        # Make sure wandb config doesn't contain any CUDA tensors
        wandb_config = {k: v.cpu().item() if isinstance(v, t.Tensor) else v for k, v in wandb_config.items()}
        wandb_process = mp.Process(
            target=new_wandb_process,
            args=(wandb_config, log_queue, wandb_entity, wandb_project),
        )
        wandb_process.start()

    # Setup save directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        # Save initial config
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"trainer": trainer.config, "run_config": run_cfg}, f, indent=2)

    # Handle normalization
    norm_factor = metadata.get("norm_factor")
    if normalize_activations and not norm_factor:
        norm_factor = get_norm_factor(data, steps=100)
        trainer.ae.scale_biases_by_name({k: 1.0 for k in norm_factor.keys()})

    # Training loop
    for step, act in enumerate(tqdm(data, total=steps, initial=start_step)):
        current_step = start_step + step

        if current_step >= steps:
            break

        act = {k: v.to(dtype=autocast_dtype, device=device) for k, v in act.items()}

        if normalize_activations and norm_factor:
            act = {k: v / norm_factor[k] for k, v in act.items()}

        # Logging
        if (use_wandb or verbose) and current_step % log_steps == 0:
            log_stats(trainer, current_step, act, log_queue=log_queue, verbose=verbose)

        # Saving checkpoints
        if save_steps is not None and current_step in save_steps and save_dir is not None:
            save_checkpoint(trainer, save_dir, current_step, norm_factor)

        # Training step
        with autocast_context:
            # Choose which model to use as primary for this step
            model_name = choice(list(act.keys()))
            trainer.update(current_step, act, model_name)

    # Save final model
    if save_dir is not None:
        save_checkpoint(trainer, save_dir, steps, norm_factor, is_final=True)

    # Clean up wandb
    if use_wandb and wandb_process:
        log_queue.put("DONE")
        wandb_process.join()

    return trainer
