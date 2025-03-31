import shutil
import os.path
import csv
import re
import os
import json
import collections
import typer
from typing import Any, Optional, Union
import logging
from pydantic import BaseModel, ValidationError
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# region Model


class Annotation(BaseModel):
    namespace: str
    data: Union[dict, list[Any]]

    class Config:
        extra = "ignore"


class FileMetadata(BaseModel):
    title: str
    duration: float

    class Config:
        extra = "ignore"


class Jams(BaseModel):
    annotations: list[Annotation]
    file_metadata: FileMetadata

    class Config:
        extra = "ignore"

    def get_annotation_by_namespace(self, namespace: str) -> Optional[str]:
        for annotation in self.annotations:
            if annotation.namespace == namespace:
                return annotation

        return None


# endregion

# region Loading


def load_jams(filepath: str) -> Optional[Jams]:
    try:
        with open(filepath, "r") as handle:
            data = json.load(handle)

        return Jams(**data)

    except FileNotFoundError:
        logger.error(f"File '{filepath}' not exists.")
    except json.JSONDecodeError:
        logger.error(f"File '{filepath}' is invalid JSON file.")
    except ValidationError as e:
        logger.error(f"Validation error during loading '{filepath}': {e}")

    return None


def scan_jams_files(directory: str):
    with os.scandir(directory) as it:
        for entry in it:
            if entry.name.endswith(".jams") and entry.is_file():
                yield entry.path

    return None


# endregion

# region Tempo


def get_tempo(jam: Jams) -> Optional[int]:
    annotation = jam.get_annotation_by_namespace("tempo")

    if annotation is not None:
        if isinstance(annotation.data, list) and len(annotation.data) > 0:
            return annotation.data[0].get("value")

    return None


def plot_tempos(tempos: list[int]) -> None:
    plt.hist(tempos, bins=20, color="royalblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Tempo (BPM)")
    plt.ylabel("Frequency")
    plt.title("Tempo Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig("./data/dataset/tempo.png", dpi=300, bbox_inches="tight")
    plt.close()


# endregion

# region KeyMode


def get_key_mode(jam: Jams) -> Optional[str]:
    annotation = jam.get_annotation_by_namespace("key_mode")

    if annotation is not None:
        if isinstance(annotation.data, list) and len(annotation.data) > 0:
            return annotation.data[0].get("value")

    return None


def plot_key_modes(key_modes: list[str]) -> None:
    plot_bar_plot(key_modes, "KeyMode Distribution", "KeyMode", "Count", "./data/dataset/keymode.png")


def plot_bar_plot(elements: list[Any], title: str, xlabel: str, ylabel: str, outfile: str):
    counter = collections.Counter(elements)
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_items)

    plt.bar(labels, values, color="royalblue", edgecolor="black", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


# endregion

# region Category


def get_category(jam: Jams) -> Optional[str]:
    pattern = r"_(\w+)\d+-"
    match = re.search(pattern, jam.file_metadata.title)

    if match:
        return match.group(1)

    return None


def plot_categories(categories: list[str]) -> None:
    plot_bar_plot(categories, "Categories Distribution", "Category", "Count", "./data/dataset/category.png")


# endregion

# region Category / Tempo


def plot_category_tempos(category_tempo: list[tuple[str, int]]) -> None:
    category_dict = collections.defaultdict(list)

    for category, tempo in category_tempo:
        category_dict[category].append(tempo)

    plt.hist(list(category_dict.values()), bins=20, edgecolor="black", label=list(category_dict.keys()), alpha=0.7)

    plt.xlabel("Tempo (BPM)")
    plt.ylabel("Frequency")
    plt.title("Tempo Distribution by Category")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig("./data/dataset/tempo_category.png", dpi=300, bbox_inches="tight")
    plt.close()


# endregion

app = typer.Typer()


@app.command()
def generate_plots():
    jams_files = scan_jams_files("./data/dataset/annotation")
    jams = [load_jams(filepath) for filepath in jams_files]
    jams = [jam for jam in jams if jam is not None]

    plot_tempos([get_tempo(jam) for jam in jams])
    plot_key_modes([get_key_mode(jam) for jam in jams])
    plot_categories([get_category(jam) for jam in jams])

    plot_category_tempos([(get_category(jam), get_tempo(jam)) for jam in jams])


@app.command()
def prepare_dataset():
    os.makedirs("./data/dataset/prepared", exist_ok=True)

    with open("./data/dataset/prepared/metadata.csv", mode="w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "tempo", "key_mode"])

        for jam_file in scan_jams_files("./data/dataset/annotation"):
            jam = load_jams(jam_file)

            if jam is None:
                continue

            category = get_category(jam)
            file_name = f"{jam.file_metadata.title}_mic.wav"

            writer.writerow([file_name, get_tempo(jam), get_key_mode(jam)])

            source_path = os.path.join("./data/dataset/audio_mono-mic", file_name)
            destination_path = os.path.join("./data/dataset/prepared/train", category, file_name)

            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy(source_path, destination_path)

    shutil.make_archive("./data/dataset/prepared/train", "zip", "./data/dataset/prepared/train")
    shutil.rmtree("./data/dataset/prepared/train")

if __name__ == "__main__":
    app()
