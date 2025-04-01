from transformers import MusicgenForConditionalGeneration, AutoProcessor
from nnsight import LanguageModel
import torch
import gc
import nnsight
from dictionary_learning import ActivationBuffer
from datasets import Dataset


class MusicGenLanguageModel(LanguageModel):
    def _load_tokenizer(self, repo_id: str, **kwargs):
        if self.tokenizer is None:
            self.tokenizer = AutoProcessor.from_pretrained(repo_id).tokenizer

    def _load_meta(
        self,
        repo_id: str,
        tokenizer_kwargs={},
        **kwargs,
    ):
        self.repo_id = repo_id

        self._load_config(repo_id, **kwargs)

        self._load_tokenizer(repo_id, **tokenizer_kwargs)
        return MusicgenForConditionalGeneration.from_pretrained(repo_id)

    def _load(
        self,
        repo_id: str,
        tokenizer_kwargs={},
        **kwargs,
    ):
        self.repo_id = repo_id

        self._load_config(repo_id, **kwargs)

        self._load_tokenizer(repo_id, **tokenizer_kwargs)
        return MusicgenForConditionalGeneration.from_pretrained(repo_id).to(kwargs["device_map"])

    @torch.no_grad()
    def generate_with_ablation(
        self,
        layer,
        prompts: list[str],
        max_tokens: int,
    ):
        with self.generate(prompts, max_new_tokens=max_tokens):
            outputs = nnsight.list().save()
            for _ in range(max_tokens):
                layer.output[0][:] = layer.input[:]
                outputs.append(self.generator.output)
                self.next()
        return outputs

    @torch.no_grad()
    def generate_clean(self, prompts: list[str], max_tokens: int):
        with self.generate(prompts, max_new_tokens=max_tokens):
            outputs = nnsight.list().save()
            for _ in range(max_tokens):
                outputs.append(self.generator.output)
                self.next()
        return outputs

    @torch.no_grad()
    def collect_generation_activations(
        self,
        layer,
        prompts: list[str],
        max_tokens: int,
    ):
        with self.generate(prompts, max_new_tokens=max_tokens):
            activations = nnsight.list().save()
            for _ in range(max_tokens):
                hidden_states = layer.output.save()
                activations.append(hidden_states[0])
                self.next()
        return torch.cat(activations)


class MusicActivationBuffer(ActivationBuffer):
    def __init__(
        self, data: Dataset, model: MusicGenLanguageModel, data_column: str, max_tokens_gen: int = 255, *args, **kwargs
    ):
        super().__init__(*args, model=model, data=data, **kwargs)
        self.max_tokens_gen = max_tokens_gen
        self.model: MusicGenLanguageModel = model
        self.data_loader = iter(self.data.batch(self.refresh_batch_size))
        self.data_column = data_column

    def text_batch(self, batch_size=None) -> list[str]:
        batch_size = batch_size if batch_size else self.refresh_batch_size
        try:
            return next(self.data_loader)
        except StopIteration:
            self.data_loader = iter(self.data.batch(batch_size))

    def refresh(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = torch.empty(
            self.activation_buffer_size, self.d_submodule, device=self.device, dtype=self.model.dtype
        )

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations
        while current_idx < self.activation_buffer_size:
            activations = self.model.collect_generation_activations(
                self.submodule, self.text_batch(), self.max_tokens_gen
            )
            remaining_space = self.activation_buffer_size - current_idx
            if remaining_space <= 0:
                break
            activations = activations[:remaining_space]
            self.activations[current_idx : current_idx + len(activations)] = activations.squeeze().to(self.device)
            current_idx += len(activations)
        self.read = torch.zeros(len(self.activations), dtype=torch.bool, device=self.device)
