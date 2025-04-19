import importlib.metadata as im
from src.project_config import INPUT_DATA_DIR
from pathlib import Path
from typing import Generator
from datasets import Dataset


def get_datasets(cfg: dict) -> Generator[tuple[Dataset, dict], None, None]:
    registry = {ep.name: ep.load() for ep in im.entry_points(group="ds_plugins")}
    for ds_conf in cfg["datasets"]:
        ds_conf["resample_sr"] = cfg["resample_sr"]
        ds_conf["seed"] = cfg["seed"]
        cls = registry[ds_conf["name"]]
        postfix = ds_conf.pop("postfix_path", None)
        if postfix:
            ds_conf["base_dir"] = INPUT_DATA_DIR / postfix
        else:
            ds_conf["base_dir"] = Path(ds_conf["base_dir"])
        yield cls(**ds_conf).prepare(**ds_conf), ds_conf
