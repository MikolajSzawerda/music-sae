from dataclasses import dataclass, field
import hydra
from datasets import load_dataset, interleave_datasets, Dataset, DatasetDict
from hydra.core.config_store import ConfigStore
from src.project_config import INPUT_DATA_DIR, MODELS_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import tqdm
import time
import torch
import joblib
import numpy as np
from collections import defaultdict
from dictionary_learning.trainers.top_k import AutoEncoderTopK


@dataclass
class TrainDatasetScriptConfig:
    regex_name: str = "*"
    split: str = "train"


@dataclass
class LinearProbingScriptConfig:
    dataset: TrainDatasetScriptConfig
    seed: int = 42
    model_name: str = "medium"
    device: str = "cuda"
    layer_id: int = 16
    activation_name: str = "raggae"
    output_name: str = "default"

    use_sae: bool = False
    sae_device: str = "cuda:0"
    sae_model_name: str = ""
    sae_batch_size: int = 32
    sae_features: list[int] = field(default_factory=list)
    sae_features_k: int = 32
    sae_generate_features: bool = False


cs = ConfigStore.instance()
cs.store(name="config_base", node=LinearProbingScriptConfig)
cs.store(name="config_sae", node=LinearProbingScriptConfig)


def debug(func):
    def wrapper(*args, **kwargs):
        enter = time.time()
        print("ENTER", func.__name__)

        result = func(*args, **kwargs)

        exit = time.time()
        elapsed = exit - enter
        print("EXIT", func.__name__, f"{elapsed:.2f}s elapsed")

        return result

    return wrapper


def add_label(batch: dict, label: int) -> dict:
    batch["label"] = [label] * len(batch["activation"])
    return batch


def unpack_and_normalize(batch: dict) -> dict:
    batch["activation"] = [item[0] for item in batch["activation"]]

    # arr = np.array(batch["activation"], dtype=float)
    # norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # batch["activation"] = (arr / norms).tolist()

    return batch


def encode_sae(model: AutoEncoderTopK, batch: dict, device: str) -> dict:
    tensor_in = torch.tensor(batch["activation"]).to(device)
    batch["activation"] = model.encode(tensor_in).detach().cpu().numpy()
    return batch


def select_features(batch: dict, features: list) -> dict:
    batch["activation"] = batch["activation"][:, features]
    return batch


def calculate_most_active_features(ds: Dataset | DatasetDict, k: int) -> dict:
    sums = dict()
    counts = defaultdict(int)

    for item in tqdm.tqdm(ds, total=(ds.num_rows)):
        label = item["label"].item()
        activation = item["activation"]

        if label not in sums:
            sums[label] = np.zeros_like(activation)

        sums[label] += activation
        counts[label] += 1

    avg = {label: sums[label] / counts[label] for label in counts.keys()}
    diff = avg[1] - avg[0]
    top_k_indices = np.argsort(np.abs(diff))[-k:][::-1]

    np.set_printoptions(threshold=np.inf)
    print(top_k_indices)

    return top_k_indices


@hydra.main(version_base=None, config_path="../conf/linear_probing", config_name="config_base")
def main(cfg: LinearProbingScriptConfig):
    ds_negative = load_dataset(
        "arrow",
        data_files=str(
            INPUT_DATA_DIR
            / f"anti_{cfg.activation_name}_activation"
            / cfg.model_name
            / cfg.dataset.regex_name
            / str(cfg.layer_id)
            / cfg.dataset.split
            / "*.arrow"
        ),
        streaming=False,
        num_proc=8,
        split="train",
    )
    ds_positive = load_dataset(
        "arrow",
        data_files=str(
            INPUT_DATA_DIR
            / f"{cfg.activation_name}_activation"
            / cfg.model_name
            / cfg.dataset.regex_name
            / str(cfg.layer_id)
            / cfg.dataset.split
            / "*.arrow"
        ),
        streaming=False,
        num_proc=8,
        split="train",
    )

    ds_negative = ds_negative.map(lambda x: add_label(x, 0), num_proc=12, batched=True, batch_size=1024)
    ds_positive = ds_positive.map(lambda x: add_label(x, 1), num_proc=12, batched=True, batch_size=1024)

    ds = interleave_datasets([ds_positive, ds_negative])
    ds = ds.map(unpack_and_normalize, num_proc=12, batched=True, batch_size=1024)
    ds.set_format(type="numpy", columns=["activation", "label"])

    if cfg.use_sae:
        sae = AutoEncoderTopK.from_pretrained(MODELS_DIR / cfg.sae_model_name).to(cfg.sae_device)
        ds = ds.map(lambda x: encode_sae(sae, x, cfg.sae_device), batched=True, batch_size=cfg.sae_batch_size)
        features = (
            calculate_most_active_features(ds, cfg.sae_features_k) if cfg.sae_generate_features else cfg.sae_features
        )
        ds = ds.map(lambda x: select_features(x, features), num_proc=12, batched=True, batch_size=cfg.sae_batch_size)

    ds.shuffle(cfg.seed)

    train_ds, test_ds = ds.train_test_split(test_size=0.1).values()

    @debug
    def split(dataset):
        X = dataset["activation"]
        y = dataset["label"]
        return X, y

    @debug
    def train(classifier, dataset):
        X_train, y_train = split(dataset)
        print(X_train[:10], y_train[:10])
        classifier.fit(X_train, y_train)

    @debug
    def evaluate(classifier, dataset):
        X_test, y_test = split(dataset)
        y_pred = classifier.predict(X_test)
        return classification_report(y_test, y_pred)

    clf = LogisticRegression(max_iter=1000)
    train(clf, train_ds)
    print(evaluate(clf, test_ds))

    save_path = MODELS_DIR / "linear_probing" / cfg.model_name / cfg.activation_name / f"{cfg.output_name}.joblib"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(clf, str(save_path))

    importance = clf.coef_[0]
    abs_importance = np.abs(importance)
    print(list(abs_importance))

    indices = np.argsort(abs_importance)[::-1]
    for i in indices[:100]:
        print(f"Feature {i}: weight={importance[i]:.4f}, abs={abs_importance[i]:.4f}")


if __name__ == "__main__":
    main()
