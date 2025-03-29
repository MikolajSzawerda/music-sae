from transformers import MusicgenForConditionalGeneration, AutoProcessor
from nnsight import LanguageModel
import nnsight
import torch

class MusicGenLanguageModel(LanguageModel):
    def _load_tokenizer(self, repo_id: str, **kwargs):
        if self.tokenizer is None:
            self.tokenizer = AutoProcessor.from_pretrained(repo_id).tokenizer

    def _load_meta(
        self,
        repo_id: str,
        tokenizer_kwargs = {},
        **kwargs,
    ):
        self.repo_id = repo_id

        self._load_config(repo_id, **kwargs)

        self._load_tokenizer(repo_id, **tokenizer_kwargs)
        return MusicgenForConditionalGeneration.from_pretrained(repo_id)
    def _load(
        self,
        repo_id: str,
        tokenizer_kwargs = {},
        **kwargs,
    ):
        self.repo_id = repo_id

        self._load_config(repo_id, **kwargs)

        self._load_tokenizer(repo_id, **tokenizer_kwargs)
        return MusicgenForConditionalGeneration.from_pretrained(repo_id).to(kwargs["device_map"])

    @torch.no_grad()
    def generate_with_ablation(self, layer, prompts: list[str], max_tokens: int, ):
        with self.generate(prompts, max_new_tokens=max_tokens):
            outputs = nnsight.list().save()
            for _ in range(max_tokens):
                layer.output[0][:] = layer.input[0][:]
                outputs.append(self.generator.output)
                self.next()
        return outputs
