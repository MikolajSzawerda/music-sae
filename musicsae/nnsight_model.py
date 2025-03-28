from transformers import MusicgenForConditionalGeneration, AutoProcessor
from nnsight import LanguageModel



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

