import argparse
import ast
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from mutagen import File as mutagen_file


from src.project_config import RAW_DATA_DIR  # type: ignore

FMA_METADATA_DIR = RAW_DATA_DIR / "fma_metadata"
FMA_MEDIUM_DIR = RAW_DATA_DIR / "fma_medium"

TRACKS_CSV = FMA_METADATA_DIR / "tracks.csv"
GENRES_CSV = FMA_METADATA_DIR / "genres.csv"
DEFAULT_OUT = FMA_MEDIUM_DIR / "medium_subset_tracks.csv"


# ---------------------------------------------------------------------------
# COMMON HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def sanitize_text(html_string: str) -> str:
    """Strip HTML tags, URLs, and duplicate spaces."""
    if pd.isna(html_string):
        return ""
    soup = BeautifulSoup(html_string, "html.parser")
    txt = soup.get_text(separator=" ", strip=True)
    txt = re.sub(r"(https?://\S+|www\.\S+)", "", txt, flags=re.I)
    return re.sub(r"\s+", " ", txt).strip()


def track_subpath(tid: int) -> str:
    """Return '000/000002.mp3' style relative path."""
    tid_str = f"{tid:06d}"
    return f"{tid_str[:3]}/{tid_str}.mp3"


def audio_duration(path: Path) -> float:
    """Length of audio file in seconds (0 if unreadable/missing)."""
    audio = mutagen_file(path)
    return 0 if audio is None or audio.info is None else audio.info.length


# ---------------------------------------------------------------------------
# PREPARE: build the medium-subset CSV
# ---------------------------------------------------------------------------
def prepare_dataset(tracks_csv: Path, genres_csv: Path, out_csv: Path) -> None:
    # ---- load & tidy the big metadata table ----
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    # literal-eval the list-like columns
    for col in [("track", "tags"), ("album", "tags"), ("artist", "tags"), ("track", "genres"), ("track", "genres_all")]:
        tracks[col] = tracks[col].map(ast.literal_eval)

    # parse the date columns
    for col in [
        ("track", "date_created"),
        ("track", "date_recorded"),
        ("album", "date_created"),
        ("album", "date_released"),
        ("artist", "date_created"),
        ("artist", "active_year_begin"),
        ("artist", "active_year_end"),
    ]:
        tracks[col] = pd.to_datetime(tracks[col])

    # categorical setup
    subset_dtype = pd.CategoricalDtype(categories=["small", "medium", "large"], ordered=True)
    tracks["set", "subset"] = tracks["set", "subset"].astype(subset_dtype)

    for col in [
        ("track", "genre_top"),
        ("track", "license"),
        ("album", "type"),
        ("album", "information"),
        ("artist", "bio"),
    ]:
        tracks[col] = tracks[col].astype("category")

    # ---- subset = medium only ----
    medium = tracks[tracks["set", "subset"] == "medium"].copy()

    # ---- load the genre id → name map ----
    genre_map = pd.read_csv(genres_csv, index_col=0)["title"].to_dict()

    def genre_names(id_list):
        return " ".join(sorted({genre_map.get(gid, "") for gid in id_list if gid in genre_map}))

    # ---- build the three output columns ----
    medium["subpath"] = medium.index.to_series().map(track_subpath)
    medium["sanitised_info"] = medium["album", "information"].map(sanitize_text)
    medium["genres_str"] = medium["track", "genres_all"].map(genre_names)
    medium["text"] = ("Genres: " + medium["genres_str"] + " " + medium["sanitised_info"]).str.strip()

    out = (
        medium[["subpath", "text"]]
        .assign(track_id=medium.index)
        .reset_index(drop=True)
        .loc[:, ["subpath", "track_id", "text"]]
    )

    out.to_csv(out_csv, index=False)
    print(f"[prepare] saved {len(out):,} rows ➜ {out_csv}")


# ---------------------------------------------------------------------------
# CHECK: drop rows whose audio is missing or too short
# ---------------------------------------------------------------------------
def check_dataset(csv_in: Path, audio_root: Path, csv_out: Path, min_seconds: int) -> None:
    df = pd.read_csv(csv_in)

    def keep(row):
        f = audio_root / row.subpath
        return f.is_file() and audio_duration(f) >= min_seconds

    mask = df.apply(keep, axis=1)
    filtered = df[mask].reset_index(drop=True)

    print(f"[check] kept {len(filtered):,} of {len(df):,} rows ({len(df) - len(filtered):,} removed)")

    filtered.to_csv(csv_out, index=False)
    print(f"[check] saved ➜ {csv_out}")


# ---------------------------------------------------------------------------
# ARGPARSE CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="FMA helper: build the medium subset CSV or remove rows with missing/short audio."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------- prepare sub-command --------
    p_prep = sub.add_parser("prepare", help="build medium_subset_tracks.csv")
    p_prep.add_argument("--tracks-csv", default=TRACKS_CSV, type=Path, help=f"tracks.csv path (default: {TRACKS_CSV})")
    p_prep.add_argument("--genres-csv", default=GENRES_CSV, type=Path, help=f"genres.csv path (default: {GENRES_CSV})")
    p_prep.add_argument("--out-csv", default=DEFAULT_OUT, type=Path, help=f"output CSV (default: {DEFAULT_OUT})")

    # -------- check sub-command --------
    p_chk = sub.add_parser("check", help="drop rows whose audio is absent or too short")
    p_chk.add_argument("--csv-in", default=DEFAULT_OUT, type=Path, help=f"input CSV (default: {DEFAULT_OUT})")
    p_chk.add_argument(
        "--audio-root",
        default=FMA_MEDIUM_DIR,
        type=Path,
        help=f"root dir containing 000/000xyz.mp3 folders (default: {FMA_MEDIUM_DIR})",
    )
    p_chk.add_argument(
        "--csv-out", default=FMA_MEDIUM_DIR / "medium_subset_tracks_5s+.csv", type=Path, help="filtered CSV output path"
    )
    p_chk.add_argument("--min-seconds", type=int, default=5, help="minimum duration in seconds (default: 5)")

    args = p.parse_args()

    if args.cmd == "prepare":
        prepare_dataset(args.tracks_csv, args.genres_csv, args.out_csv)
    elif args.cmd == "check":
        check_dataset(args.csv_in, args.audio_root, args.csv_out, args.min_seconds)


if __name__ == "__main__":
    main()
