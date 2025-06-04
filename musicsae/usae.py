import torch as t
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dictionary_learning.dictionary import Dictionary
from dictionary_learning.trainers.trainer import (
    set_decoder_norm_to_unit_norm,
    SAETrainer,
    get_lr_schedule,
    remove_gradient_parallel_to_decoder_directions,
)
from dictionary_learning.trainers.top_k import geometric_median
from collections import namedtuple


class BufferDict(nn.Module):
    def __init__(self, mapping: dict[str, t.Tensor] | None = None):
        super().__init__()
        mapping = {} if mapping is None else mapping
        for k, v in mapping.items():
            self[k] = v

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, tensor):
        self.register_buffer(key, tensor)

    def __iter__(self):
        return iter(self._buffers)

    def items(self):
        return self._buffers.items()


class UniversalAutoEncoder(Dictionary, nn.Module):
    def __init__(self, model_activation_dims: Dict[str, int], dict_size: int, k: int):
        super().__init__()
        self.model_activation_dims = model_activation_dims
        self.dict_size = dict_size
        self.activation_dim = model_activation_dims
        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.thresholds = BufferDict(
            {f"{model_name}": t.tensor(-1.0, dtype=t.float32) for model_name in model_activation_dims.keys()}
        )

        self.encoders = nn.ModuleDict(
            {model_name: nn.Linear(dim, dict_size) for model_name, dim in model_activation_dims.items()}
        )

        self.decoders = nn.ModuleDict(
            {model_name: nn.Linear(dict_size, dim, bias=False) for model_name, dim in model_activation_dims.items()}
        )

        for decoder in self.decoders.values():
            decoder.weight.data = set_decoder_norm_to_unit_norm(
                decoder.weight,
                decoder.weight.shape[0],  # output dim
                decoder.weight.shape[1],  # input dim
            )

        for model_name in model_activation_dims.keys():
            self.encoders[model_name].weight.data = self.decoders[model_name].weight.T.clone()
            self.encoders[model_name].bias.data.zero_()

        self.b_dec = nn.ParameterDict(
            {model_name: nn.Parameter(t.zeros(dim)) for model_name, dim in model_activation_dims.items()}
        )

    def encode(
        self, x: t.Tensor, model_name: str, return_topk: bool = False, use_threshold: bool = False
    ) -> Tuple[t.Tensor, ...]:
        post_relu_feat_acts = nn.functional.relu(self.encoders[model_name](x - self.b_dec[model_name]))

        if use_threshold:
            encoded_acts = post_relu_feat_acts * (post_relu_feat_acts > self.thresholds[f"{model_name}"])
            if return_topk:
                post_topk = post_relu_feat_acts.topk(self.k, sorted=False, dim=-1)
                return encoded_acts, post_topk.values, post_topk.indices, post_relu_feat_acts
            else:
                return encoded_acts

        post_topk = post_relu_feat_acts.topk(self.k, sorted=False, dim=-1)

        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = t.zeros_like(post_relu_feat_acts)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK, post_relu_feat_acts
        else:
            return encoded_acts_BF

    def decode(self, z: t.Tensor, model_name: str) -> t.Tensor:
        return self.decoders[model_name](z) + self.b_dec[model_name]

    # TODO parralelize this
    def decode_all(self, z: t.Tensor) -> Dict[str, t.Tensor]:
        return {model_name: self.decode(z, model_name) for model_name in self.decoders.keys()}

    def forward(
        self, x: t.Tensor, model_name: str, output_features: bool = False, output_all: bool = False
    ) -> Tuple[t.Tensor, ...]:
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
        for model_name in self.model_activation_dims.keys():
            self.encoders[model_name].bias.data *= scale
            self.b_dec[model_name].data *= scale
            if self.thresholds[model_name] >= 0:
                self.thresholds[model_name] *= scale

    def scale_biases_by_name(self, scales: dict[str, float]):
        for model_name, scale in scales.items():
            self.encoders[model_name].bias.data *= scale
            self.b_dec[model_name].data *= scale
            if self.thresholds[model_name] >= 0:
                self.thresholds[model_name] *= scale

    @classmethod
    def from_pretrained(cls, path: str, k: Optional[int] = None, device: Optional[str] = None):
        state_dict = t.load(path)

        # Extract model dimensions from state dict
        model_activation_dims = {}
        for key in state_dict.keys():
            if key.startswith("decoders."):
                model_name = key.split(".")[1]
                if model_name not in model_activation_dims:
                    model_activation_dims[model_name] = state_dict[key].shape[0]

        dict_size = state_dict["encoders." + list(model_activation_dims.keys())[0] + ".weight"].shape[0]

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
        return (
            list(self.encoders[model_name].parameters())
            + list(self.decoders[model_name].parameters())
            + [self.b_dec[model_name]]
        )


class USAETopKTrainer(SAETrainer):
    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dims: Dict[str, int],
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        dict_class: type = UniversalAutoEncoder,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,  # see Appendix A.2
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
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

        # Initialise autoencoder
        self.ae = dict_class(activation_dims, dict_size, k)
        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)
        self.top_k_auxs, self.num_tokens_since_fired, self.optimizers, self.schedulers = {}, {}, {}, {}
        for name in activation_dims.keys():
            self.top_k_auxs[name] = activation_dims[name] // 2
            self.num_tokens_since_fired[name] = t.zeros(dict_size, dtype=t.long, device=device)
            self.optimizers[name] = t.optim.Adam(self.ae.get_model_parameters(name), lr=self.lr, betas=(0.9, 0.999))
            self.schedulers[name] = t.optim.lr_scheduler.LambdaLR(self.optimizers[name], lr_lambda=lr_fn)

        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

    def get_auxiliary_loss(self, num_tokens_since_fired, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor, top_k_aux):
        dead_features = num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(top_k_aux, self.dead_features)

            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(residual_BD.shape)
            loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, top_acts_BK: t.Tensor, model_name: str):
        device_type = "cuda" if top_acts_BK.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            active = top_acts_BK.clone().detach()
            active[active <= 0] = float("inf")
            min_activations = active.min(dim=1).values.to(dtype=t.float32)
            min_activation = min_activations.mean()

            B, K = active.shape
            assert len(active.shape) == 2
            assert min_activations.shape == (B,)

            if self.ae.thresholds[model_name] < 0:
                self.ae.thresholds[model_name] = min_activation
            else:
                self.ae.thresholds[model_name] = (self.threshold_beta * self.ae.thresholds[model_name]) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def loss(self, x: dict[str, t.Tensor], model_name: str, step=None, logging=False):
        # Run the SAE
        f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
            x[model_name], model_name, return_topk=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            self.update_threshold(top_acts_BK, model_name)

        x_hat = self.ae.decode_all(f)
        e = x[model_name] - x_hat[model_name]
        num_tokens_in_step = x[model_name].size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired[model_name], dtype=t.bool)
        did_fire[top_indices_BK.flatten()] = True
        self.num_tokens_since_fired[model_name] += num_tokens_in_step
        self.num_tokens_since_fired[model_name][did_fire] = 0
        auxk_loss = (
            self.get_auxiliary_loss(
                self.num_tokens_since_fired[model_name], e.detach(), post_relu_acts_BF, self.top_k_auxs[model_name]
            )
            if self.auxk_alpha > 0
            else 0
        )
        l2_loss = 0

        for name in self.activation_dims.keys():
            e = x[name] - x_hat[name]
            l2_loss += e.pow(2).sum(dim=-1).mean()
        loss = l2_loss + self.auxk_alpha * auxk_loss
        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x[model_name],
                x_hat[model_name],
                f,
                {"l2_loss": l2_loss.item(), "auxk_loss": auxk_loss.item(), "loss": loss.item()},
            )

    def update(self, step, x: dict[str, t.Tensor], model_name: str):
        # Initialise the decoder bias
        decoder, optimizer, scheduler = (
            self.ae.decoders[model_name],
            self.optimizers[model_name],
            self.schedulers[model_name],
        )
        activation_dim = self.activation_dims[model_name]
        if step == 0:
            for name in self.activation_dims.keys():
                median = geometric_median(x[name])
                median = median.to(self.ae.b_dec[name].dtype)
                self.ae.b_dec[name].data = median

        # compute the loss
        x = {name: val.to(self.device) for name, val in x.items()}
        loss = self.loss(x, model_name, step=step)
        loss.backward()

        # clip grad norm and remove grads parallel to decoder directions
        decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            decoder.weight,
            decoder.weight.grad,
            activation_dim,
            self.ae.dict_size,
        )
        t.nn.utils.clip_grad_norm_(self.ae.get_model_parameters(model_name), 1.0)

        # do a training step
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Make sure the decoder is still unit-norm
        decoder.weight.data = set_decoder_norm_to_unit_norm(decoder.weight, activation_dim, self.ae.dict_size)

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "USAETopKTrainer",
            "dict_class": "UniversalAutoEncoder",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }
