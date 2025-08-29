import tqdm
from src.project_config import INPUT_DATA_DIR


def read_histogram() -> list[float]:
    with open("data/tmp/bpm_hist.txt", "r") as handle:
        data = handle.read()

    data = data.replace("[", "").replace("]", "")
    lines = data.split("\n")
    lines = [line for line in lines if line != ""]
    values = [float(v) for v in lines]

    return values


def get_tracks() -> list[str]:
    with open("data/input/mtg-jamendo/tracks_bpm.csv", "r") as handle:
        lines = handle.read().split("\n")

    header = lines[0]
    lines = lines[1:]
    output_lines = []

    for line in lines:
        path = line.split(",")[0]
        path = INPUT_DATA_DIR / "mtg-jamendo" / "datashare-instruments" / path

        if path.exists():
            output_lines.append(line)

    return header, output_lines


def main():
    hist = read_histogram()
    header, tracks = get_tracks()

    with (
        open("data/input/mtg-jamendo/tracks_slow.csv", "w") as slow,
        open("data/input/mtg-jamendo/tracks_fast.csv", "w") as fast,
    ):
        slow.write(f"{header}\n")
        fast.write(f"{header}\n")

        for bpm, track in tqdm.tqdm(zip(hist, tracks)):
            if bpm >= THRESHOLD:
                fast.write(f"{track}\n")
            else:
                slow.write(f"{track}\n")


if __name__ == "__main__":
    THRESHOLD = 117
    main()
