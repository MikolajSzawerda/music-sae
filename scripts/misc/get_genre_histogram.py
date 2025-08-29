import csv
from tqdm import tqdm

all_genres = {}

with open("data/input/mtg-jamendo/tracks.csv", "r") as handle:
    reader = csv.reader(handle)

    next(reader)

    for row in tqdm(reader):
        genres = row[2]

        if "Music with genres:" in genres:
            genres = genres[genres.index("Music with genres:") + len("Music with genres:") :]

        if "having instruments:" in genres:
            genres = genres[: genres.index("having instruments:")]

        if "in the mood of:" in genres:
            genres = genres[: genres.index("in the mood of:")]

        genres = [g.strip() for g in genres.split(",")]

        for g in genres:
            if g not in all_genres:
                all_genres[g] = 0

            all_genres[g] += 1

for genre, count in sorted(all_genres.items(), key=lambda k: k[1]):
    print(genre, ":", count)
