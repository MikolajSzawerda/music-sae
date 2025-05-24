import pandas as pd
import re
from bs4 import BeautifulSoup
import ast
from src.project_config import RAW_DATA_DIR


def load(filepath):
    tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

    COLUMNS = [("track", "tags"), ("album", "tags"), ("artist", "tags"), ("track", "genres"), ("track", "genres_all")]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [
        ("track", "date_created"),
        ("track", "date_recorded"),
        ("album", "date_created"),
        ("album", "date_released"),
        ("artist", "date_created"),
        ("artist", "active_year_begin"),
        ("artist", "active_year_end"),
    ]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ("small", "medium", "large")
    try:
        tracks["set", "subset"] = tracks["set", "subset"].astype("category", categories=SUBSETS, ordered=True)
    except (ValueError, TypeError):
        # the categories and ordered arguments were removed in pandas 0.25
        tracks["set", "subset"] = tracks["set", "subset"].astype(pd.CategoricalDtype(categories=SUBSETS, ordered=True))

    COLUMNS = [
        ("track", "genre_top"),
        ("track", "license"),
        ("album", "type"),
        ("album", "information"),
        ("artist", "bio"),
    ]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype("category")

    return tracks


def sanitize_text(html_string: str) -> str:
    """Remove HTML, URLs, duplicate spaces, etc."""
    if pd.isna(html_string):
        return ""
    soup = BeautifulSoup(html_string, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"(https?://\S+|www\.\S+)", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def track_subpath(tid: int) -> str:
    """Return '000/000002.mp3'-style sub-path for a given track id."""
    tid_str = f"{tid:06d}"
    return f"{tid_str[:3]}/{tid_str}.mp3"


# ------------------------------------------------------------------
# 2. load metadata
# ------------------------------------------------------------------
tracks = load(RAW_DATA_DIR / "fma_metadata" / "tracks.csv")
genres = pd.read_csv(RAW_DATA_DIR / "fma_metadata" / "genres.csv", index_col=0)

# id ➜ genre-title map (there is always a 'title' column in genres.csv)
genre_map = genres["title"].to_dict()

# ------------------------------------------------------------------
# 3. keep only the ‘medium’ subset and build the three output columns
# ------------------------------------------------------------------
medium = tracks[tracks["set", "subset"] == "medium"].copy()

# -- column: subpath
medium["subpath"] = medium.index.to_series().map(track_subpath)

# -- column: sanitised album info
medium["sanitised_info"] = medium["album", "information"].map(sanitize_text)


# -- column: genre names as a single space-separated string
def genre_names(id_list):
    return " ".join(sorted({genre_map.get(gid, "") for gid in id_list if gid in genre_map}))


medium["genres_str"] = medium["track", "genres_all"].map(genre_names)

# -- final text column = info + genres
# medium["text"] = (medium["sanitised_info"] + " " + medium["genres_str"]).str.strip()
medium["text"] = ("Genres: " + medium["genres_str"] + " " + medium["sanitised_info"]).str.strip()
# ------------------------------------------------------------------
# 4. select & write
# ------------------------------------------------------------------
out = (
    medium.loc[:, ["subpath", "text"]]  # keep the new columns
    .assign(track_id=medium.index)  # move index into a column
    .reset_index(drop=True)
    .loc[:, ["subpath", "track_id", "text"]]  # desired order
)

out.to_csv(RAW_DATA_DIR / "fma_medium" / "medium_subset_tracks.csv", index=False)
print(f"Saved {len(out):,} rows ➜ medium_subset_tracks.csv")
