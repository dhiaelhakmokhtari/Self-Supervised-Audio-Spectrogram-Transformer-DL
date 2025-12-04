import os
import csv
import json
import random
from glob import glob
from collections import defaultdict

DATA_DIR = "./data/gtzan_10s"
OUTPUT_DIR = "metadata"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train/val/test


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "song_id"])
        writer.writerows(rows)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ==============================================
# STEP 1 — Load dataset and group by song_id
# ==============================================
genres = sorted(os.listdir(DATA_DIR))

dataset = {}  # genre -> { song_id : [file1, file2, file3] }

for genre in genres:
    genre_dir = os.path.join(DATA_DIR, genre)

    song_groups = defaultdict(list)

    for fp in glob(os.path.join(genre_dir, "*.wav")):
        base = os.path.basename(fp)

        # blues.00024_1.wav --> blues.00024
        song_id = base.split("_")[0]

        song_groups[song_id].append(fp)

    dataset[genre] = song_groups


# ==============================================
# STEP 2 — Split song groups (not individual files)
# ==============================================
train_rows, val_rows, test_rows = [], [], []
train_json, val_json, test_json = [], [], []

for genre, groups in dataset.items():

    song_ids = list(groups.keys())
    random.shuffle(song_ids)

    N = len(song_ids)
    n_train = int(N * SPLIT_RATIOS[0])
    n_val   = int(N * SPLIT_RATIOS[1])
    n_test  = N - n_train - n_val

    train_ids = song_ids[:n_train]
    val_ids   = song_ids[n_train:n_train+n_val]
    test_ids  = song_ids[n_train+n_val:]

    # helper to flatten into csv/json
    def push(ids, rows, jrows):
        for sid in ids:
            for fp in sorted(groups[sid]):  # sorted for clean output
                rows.append([fp, genre, sid])
                jrows.append({"wav": fp, "labels": genre})

    push(train_ids, train_rows, train_json)
    push(val_ids,   val_rows,   val_json)
    push(test_ids,  test_rows,  test_json)


# ==============================================
# STEP 3 — Save outputs
# ==============================================
write_csv(os.path.join(OUTPUT_DIR, "train.csv"), train_rows)
write_csv(os.path.join(OUTPUT_DIR, "val.csv"),   val_rows)
write_csv(os.path.join(OUTPUT_DIR, "test.csv"),  test_rows)

write_json(os.path.join(OUTPUT_DIR, "train.json"), train_json)
write_json(os.path.join(OUTPUT_DIR, "val.json"),   val_json)
write_json(os.path.join(OUTPUT_DIR, "test.json"),  test_json)