from abc import ABC, abstractmethod
from datasets import Dataset


class AudioDatasetPlugin(ABC):
    name: str  # unique id – becomes the entry‑point name

    @abstractmethod
    def load(self, split: str, **kw) -> Dataset:  # download/stream
        ...

    def prepare(self, split: str = "train", **kw) -> Dataset:
        """Default pipeline: load ➜ custom row ops ➜ return HF Dataset."""
        ds = self.load(split, **kw)
        return self.apply_transforms(ds)

    # override only if you need per‑dataset transforms
    def apply_transforms(self, ds: Dataset) -> Dataset:
        return ds
