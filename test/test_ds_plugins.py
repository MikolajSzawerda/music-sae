from src.dataset_plugins import get_datasets
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="../conf/collect", config_name="config-new")
def main(args: DictConfig):
    ds_iter = get_datasets(OmegaConf.to_container(args, resolve=True))
    print(next(ds_iter))
    print(next(ds_iter))
    print(next(ds_iter))


if __name__ == "__main__":
    main()
