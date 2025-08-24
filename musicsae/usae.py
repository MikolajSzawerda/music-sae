import torch as t
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dictionary_learning.dictionary import Dictionary
from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer, geometric_median
from dictionary_learning.trainers.trainer import SAETrainer
from collections import namedtuple
from random import choice


class UniversalAutoEncoder(Dictionary, nn.Module):
    """
    Universal SAE that can encode/decode activations from multiple models.
    This is essentially a collection of TopK SAEs that share the same feature space.
    """

    def __init__(self, model_activation_dims: Dict[str, int], dict_size: int, k: int):
        super().__init__()
        self.model_activation_dims = model_activation_dims
        self.dict_size = dict_size
        self.activation_dim = model_activation_dims  # Keep for compatibility
        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))

        # Create individual TopK SAEs for each model
        self.saes = nn.ModuleDict()
        for model_name, activation_dim in model_activation_dims.items():
            self.saes[model_name] = AutoEncoderTopK(activation_dim, dict_size, k)

    def encode(
        self, x: t.Tensor, model_name: str, return_topk: bool = False, use_threshold: bool = False
    ) -> Tuple[t.Tensor, ...]:
        """Encode activations from the specified model."""
        return self.saes[model_name].encode(x, return_topk=return_topk, use_threshold=use_threshold)

    def decode(self, z: t.Tensor, model_name: str) -> t.Tensor:
        """Decode features to activations for the specified model."""
        return self.saes[model_name].decode(z)

    def decode_all(self, z: t.Tensor) -> Dict[str, t.Tensor]:
        """Decode features to activations for all models."""
        return {model_name: sae.decode(z) for model_name, sae in self.saes.items()}

    def forward(
        self, x: t.Tensor, model_name: str, output_features: bool = False, output_all: bool = False
    ) -> Tuple[t.Tensor, ...]:
        """Forward pass through the SAE."""
        encoded_acts = self.encode(x, model_name)

        if output_all:
            x_hat = self.decode_all(encoded_acts)
        else:
            x_hat = self.decode(encoded_acts, model_name)

        if not output_features:
            return x_hat
        else:
            return x_hat, encoded_acts

    def scale_biases(self, scale: float):
        """Scale biases for all models by the same factor."""
        for sae in self.saes.values():
            sae.scale_biases(scale)

    def scale_biases_by_name(self, scales: Dict[str, float]):
        """Scale biases for each model by different factors."""
        for model_name, scale in scales.items():
            if model_name in self.saes:
                self.saes[model_name].scale_biases(scale)

    @classmethod
    def from_pretrained(cls, path: str, k: Optional[int] = None, device: Optional[str] = None):
        """
        Load a pretrained Universal SAE model from file (for inference only).

        For resuming training, use USAETopKTrainer.load_trainer_state() instead.
        """
        checkpoint_data = t.load(path, map_location=device)

        # Check if this is a trainer checkpoint or just model weights
        if "ae_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["ae_state_dict"]
        else:
            state_dict = checkpoint_data

        # Extract model dimensions from state dict
        model_activation_dims = {}
        for key in state_dict.keys():
            if key.startswith("saes.") and key.endswith(".decoder.weight"):
                model_name = key.split(".")[1]
                if model_name not in model_activation_dims:
                    model_activation_dims[model_name] = state_dict[key].shape[0]

        # Get dict_size from any encoder
        first_model = list(model_activation_dims.keys())[0]
        dict_size = state_dict[f"saes.{first_model}.encoder.weight"].shape[0]

        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = cls(model_activation_dims, dict_size, k)
        autoencoder.load_state_dict(state_dict)

        if device is not None:
            autoencoder.to(device)

        return autoencoder

    def get_model_parameters(self, model_name: str):
        """Get parameters for a specific model's SAE."""
        return list(self.saes[model_name].parameters())


class USAETopKTrainer(SAETrainer):
    """
    Trainer for Universal SAE that manages multiple TopK trainers.
    Much cleaner implementation using composition.
    """

    def __init__(
        self,
        steps: int,
        activation_dims: Dict[str, int],
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        dict_class: type = UniversalAutoEncoder,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "UniversalAutoEncoder",
        submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.k = k
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step
        self.activation_dims = activation_dims

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Initialize Universal SAE
        self.ae = dict_class(activation_dims, dict_size, k)
        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        # Set learning rate
        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000

        # Create individual trainers for each model
        self.trainers = {}
        for model_name, activation_dim in activation_dims.items():
            # Create a TopK trainer for each model
            trainer = TopKTrainer(
                steps=steps,
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                layer=layer,
                lm_name=f"{lm_name}_{model_name}",
                dict_class=AutoEncoderTopK,
                lr=lr,
                auxk_alpha=auxk_alpha,
                warmup_steps=warmup_steps,
                decay_start=decay_start,
                threshold_beta=threshold_beta,
                threshold_start_step=threshold_start_step,
                seed=seed,
                device=device,
                wandb_name=f"{wandb_name}_{model_name}",
                submodule_name=submodule_name,
            )
            # Replace the trainer's ae with our shared SAE component
            trainer.ae = self.ae.saes[model_name]

            # CRITICAL: Re-create optimizer and scheduler with correct parameters after SAE replacement
            from dictionary_learning.trainers.trainer import get_lr_schedule

            trainer.optimizer = t.optim.Adam(trainer.ae.parameters(), lr=trainer.lr, betas=(0.9, 0.999))
            lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)
            trainer.scheduler = t.optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=lr_fn)

            self.trainers[model_name] = trainer

        # Inherit logging parameters from the first trainer
        first_trainer = next(iter(self.trainers.values()))
        self.logging_parameters = first_trainer.logging_parameters

        # Store current model for standard interface compatibility
        self._current_model = None

    def loss(self, x: Dict[str, t.Tensor], model_name: str = None, step=None, logging=False):
        """Compute loss using the specified model's trainer but with cross-model reconstruction."""
        # If no model specified, choose one
        if model_name is None:
            if isinstance(x, dict):
                model_name = choice(list(x.keys()))
            else:
                # If single tensor input, use the stored current model or first available
                model_name = self._current_model or list(self.trainers.keys())[0]

        # Store for future reference
        self._current_model = model_name

        # Get the primary model's trainer
        trainer = self.trainers[model_name]

        # Run encoding with the primary model
        f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
            x[model_name], model_name, return_topk=True, use_threshold=False
        )

        # Update threshold using the primary model's trainer logic
        if step is not None and step > trainer.threshold_start_step:
            trainer.update_threshold(top_acts_BK)

        # Decode to ALL models (this is the key universal capability)
        x_hat = self.ae.decode_all(f)

        sub_losses = {}
        # Compute L2 loss across all models (only for models present in input)
        # l2_loss = 0
        # for name in self.activation_dims.keys():
        #     if name in x:  # Only compute loss for models present in input
        #         e = x[name] - x_hat[name]
        #         multix = 10 if 'medium' in name else 1
        #         l2_loss += multix * e.pow(2).sum(dim=-1).mean()
        l2_loss = 0.0

        for name in self.activation_dims.keys():
            if name in x:  # Only compute loss for models present in input
                e = x[name] - x_hat[name]
                multix = 1.5 if ("medium" in name and (step and step % 2 == 0 and not logging)) else 1
                # multix = 5
                sub_loss = multix * e.abs().sum(dim=-1).mean()
                sub_losses[f"{model_name}_{name}_loss"] = sub_loss.item()
                l2_loss += sub_loss

        # Update effective L0 (should be K)
        trainer.effective_l0 = top_acts_BK.size(1)

        # Update "number of tokens since fired" for each feature
        num_tokens_in_step = x[model_name].size(0)
        did_fire = t.zeros_like(trainer.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices_BK.flatten()] = True
        trainer.num_tokens_since_fired += num_tokens_in_step
        trainer.num_tokens_since_fired[did_fire] = 0

        # Compute auxiliary loss using primary model's logic
        primary_e = x[model_name] - x_hat[model_name]

        auxk_loss = trainer.get_auxiliary_loss(primary_e.detach(), post_relu_acts_BF) if trainer.auxk_alpha > 0 else 0

        loss = l2_loss + trainer.auxk_alpha * auxk_loss
        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {"l2_loss": l2_loss.item(), "auxk_loss": auxk_loss.item(), "loss": loss.item()} | sub_losses,
            )

    def update(self, step, x, model_name: str = None):
        """
        Update method compatible with standard SAE trainer interface.
        Supports both single-model and multi-model input.
        """
        # Handle both single tensor and dict inputs
        if not isinstance(x, dict):
            # Single tensor input - need to figure out which model this belongs to
            if model_name is None:
                model_name = self._current_model or list(self.trainers.keys())[0]
            x = {model_name: x}
        elif model_name is None:
            # Dict input but no model specified - choose one
            model_name = choice(list(x.keys()))

        # Store for future reference
        self._current_model = model_name

        # Initialize decoder bias on first step
        if step == 0:
            for name in self.activation_dims.keys():
                if name in x:  # Only initialize if we have data for this model
                    median = geometric_median(x[name])
                    median = median.to(self.ae.saes[name].b_dec.dtype)
                    self.ae.saes[name].b_dec.data = median

        # Move data to device
        x = {name: val.to(self.device) for name, val in x.items()}

        # Get the trainer for the primary model
        trainer = self.trainers[model_name]

        # Compute loss
        loss = self.loss(x, model_name, step=step)
        loss.backward()

        # Apply gradients and constraints to the primary model's SAE
        from dictionary_learning.trainers.trainer import (
            remove_gradient_parallel_to_decoder_directions,
            set_decoder_norm_to_unit_norm,
        )

        trainer.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            trainer.ae.decoder.weight,
            trainer.ae.decoder.weight.grad,
            trainer.ae.activation_dim,
            trainer.ae.dict_size,
        )
        t.nn.utils.clip_grad_norm_(trainer.ae.parameters(), 1.0)

        # Update using primary model's optimizer
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        trainer.scheduler.step()

        # Ensure decoder remains unit-norm
        trainer.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            trainer.ae.decoder.weight, trainer.ae.activation_dim, trainer.ae.dict_size
        )

        return loss.item()

    def update_multi_model(self, step, x: Dict[str, t.Tensor], model_name: str):
        """Explicit multi-model update method (for backward compatibility)."""
        return self.update(step, x, model_name)

    def get_logging_parameters(self):
        """Get logging parameters from the most recently used trainer."""
        # Return from the current/primary trainer if available
        if self._current_model and self._current_model in self.trainers:
            return self.trainers[self._current_model].get_logging_parameters()
        else:
            # Fallback to first trainer
            first_trainer = next(iter(self.trainers.values()))
            return first_trainer.get_logging_parameters()

    def save_trainer_state(self):
        """Save complete trainer state including all sub-trainers."""
        state = {"ae_state_dict": self.ae.state_dict(), "trainer_config": self.config, "trainer_states": {}}

        # Save state of each sub-trainer
        for model_name, trainer in self.trainers.items():
            state["trainer_states"][model_name] = {
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": trainer.scheduler.state_dict(),
                "num_tokens_since_fired": trainer.num_tokens_since_fired.clone(),
                "threshold": trainer.ae.threshold.clone() if hasattr(trainer.ae, "threshold") else None,
                "effective_l0": trainer.effective_l0,
                "dead_features": trainer.dead_features,
                "pre_norm_auxk_loss": trainer.pre_norm_auxk_loss,
            }

        return state

    @classmethod
    def load_trainer_state(cls, state_dict, device="cuda"):
        """Load complete trainer state from saved state."""
        # Extract trainer config
        trainer_config = state_dict["trainer_config"]

        # Create trainer with config (excluding class names)
        init_config = {
            k: v for k, v in trainer_config.items() if k not in ["trainer_class", "dict_class", "activation_dim"]
        }
        init_config["device"] = device

        # Handle legacy configs that might have activation_dim instead of activation_dims
        if "activation_dims" not in init_config and "activation_dim" in trainer_config:
            # This is likely a corrupted config, we'll need to reconstruct from the model state
            print("Warning: Legacy config detected, reconstructing activation_dims from model state...")
            ae_state = state_dict["ae_state_dict"]
            activation_dims = {}
            for key in ae_state.keys():
                if key.startswith("saes.") and key.endswith(".decoder.weight"):
                    model_name = key.split(".")[1]
                    if model_name not in activation_dims:
                        activation_dims[model_name] = ae_state[key].shape[0]
            init_config["activation_dims"] = activation_dims

        trainer = cls(**init_config)

        # Load model state
        trainer.ae.load_state_dict(state_dict["ae_state_dict"])
        trainer.ae.to(device)

        # Restore each sub-trainer's state
        if "trainer_states" in state_dict:
            for model_name, sub_state in state_dict["trainer_states"].items():
                if model_name in trainer.trainers:
                    sub_trainer = trainer.trainers[model_name]

                    # Restore optimizer and scheduler states
                    sub_trainer.optimizer.load_state_dict(sub_state["optimizer_state_dict"])
                    sub_trainer.scheduler.load_state_dict(sub_state["scheduler_state_dict"])

                    # Restore training tracking variables
                    sub_trainer.num_tokens_since_fired = sub_state["num_tokens_since_fired"].to(device)
                    if sub_state["threshold"] is not None:
                        sub_trainer.ae.threshold = sub_state["threshold"].to(device)
                    sub_trainer.effective_l0 = sub_state["effective_l0"]
                    sub_trainer.dead_features = sub_state["dead_features"]
                    sub_trainer.pre_norm_auxk_loss = sub_state["pre_norm_auxk_loss"]

        return trainer

    @property
    def config(self):
        """Configuration for the Universal SAE trainer."""
        return {
            "trainer_class": "USAETopKTrainer",
            "dict_class": "UniversalAutoEncoder",
            "steps": self.steps,
            "activation_dims": self.activation_dims,
            "dict_size": self.ae.dict_size,
            "k": self.k,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "lr": self.lr,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "seed": self.seed,
            "device": self.device,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }
