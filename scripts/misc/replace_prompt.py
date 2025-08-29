import csv
import tqdm


HEADER = ["path", "track_id", "text"]
PROMPT = [""]

with open("data/input/mtg-jamendo/tracks_anti_dumb4.csv", "r") as handle:
    reader = csv.reader(handle)
    next(reader)

    rows = [row for row in tqdm.tqdm(reader)]

with open("data/input/mtg-jamendo/tracks_anti_dumb4.csv", "w") as handle:
    writer = csv.writer(handle)
    writer.writerow(HEADER)

    for row in tqdm.tqdm(rows):
        row[2] = "Music with genres: jazz"
        writer.writerow(row)
