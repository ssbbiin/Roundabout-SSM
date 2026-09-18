from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path.home() / "roundabout_ssm"

DATA = ROOT / "data/raw/data"
EVENTS = ROOT / "data/processed/location0_entry_crossings.csv"
SPLIT_FILE = ROOT / "configs/recording_split.json"

OUT_DIR = ROOT / "data/splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_SEC = 2.0

MIN_TTE = 0.4
MAX_TTE = 4.0
SAMPLE_INTERVAL = 0.2

GO_THRESHOLD = 1.6


with open(SPLIT_FILE, "r") as f:
    split_cfg = json.load(f)

split_map = {}

for split_name in ["train", "val", "test"]:
    for rec in split_cfg[split_name]:
        split_map[int(rec)] = split_name


events = pd.read_csv(EVENTS)

samples = []

print("=" * 100)
print("BUILD DECISION SAMPLE INDEX")
print("=" * 100)

tte_values = np.round(
    np.arange(
        MIN_TTE,
        MAX_TTE + 1e-9,
        SAMPLE_INTERVAL
    ),
    2
)

print("TTE candidates:", tte_values.tolist())
print()


for rec_id, rec_events in events.groupby("recordingId"):

    rec_id = int(rec_id)
    rid = f"{rec_id:02d}"

    if rec_id not in split_map:
        continue

    print(
        f"Processing recording {rid} "
        f"[{split_map[rec_id]}] ..."
    )

    recording_meta = pd.read_csv(
        DATA / f"{rid}_recordingMeta.csv"
    ).iloc[0]

    frame_rate = float(recording_meta["frameRate"])

    track_meta = pd.read_csv(
        DATA / f"{rid}_tracksMeta.csv"
    )

    initial_frame = dict(
        zip(
            track_meta["trackId"].astype(int),
            track_meta["initialFrame"].astype(int)
        )
    )

    final_frame = dict(
        zip(
            track_meta["trackId"].astype(int),
            track_meta["finalFrame"].astype(int)
        )
    )

    for _, event in rec_events.iterrows():

        track_id = int(event["trackId"])

        if track_id not in initial_frame:
            continue

        cross_frame = int(event["crossFrame"])

        for tte in tte_values:

            frame_offset = int(
                round(tte * frame_rate)
            )

            current_frame = (
                cross_frame - frame_offset
            )

            history_frames = int(
                round(HISTORY_SEC * frame_rate)
            )

            history_start = (
                current_frame - history_frames + 1
            )

            # Ego history 전체 2초가 존재해야 함
            if history_start < initial_frame[track_id]:
                continue

            if current_frame > final_frame[track_id]:
                continue

            decision = (
                1 if tte <= GO_THRESHOLD else 0
            )

            samples.append({
                "split": split_map[rec_id],
                "recordingId": rec_id,
                "trackId": track_id,
                "class": event["class"],
                "entryId": int(event["entryId"]),

                "currentFrame": int(current_frame),
                "historyStartFrame": int(history_start),
                "crossFrame": int(cross_frame),

                "timeToEntry": float(tte),
                "decision": int(decision),

                "entrySpeed": float(event["speed"]),
                "entryHeading": float(event["heading"]),

                "frameRate": frame_rate
            })


df = pd.DataFrame(samples)

print()
print("=" * 100)
print("DATASET SUMMARY")
print("=" * 100)

print(f"Total samples : {len(df)}")
print(
    "Unique vehicles:",
    df[
        ["recordingId", "trackId"]
    ].drop_duplicates().shape[0]
)

print("\n=== BY SPLIT ===")
print(df["split"].value_counts())

print("\n=== GO / WAIT BY SPLIT ===")

table = pd.crosstab(
    df["split"],
    df["decision"],
    margins=True
)

table = table.rename(
    columns={
        0: "WAIT",
        1: "GO"
    }
)

print(table)


print("\n=== GO RATE ===")

for name in ["train", "val", "test"]:

    d = df[df["split"] == name]

    go_rate = (
        100.0 * d["decision"].mean()
    )

    print(
        f"{name:5s}: "
        f"{len(d):6d} samples | "
        f"GO {go_rate:6.2f}% | "
        f"WAIT {100-go_rate:6.2f}%"
    )


print("\n=== TTE DISTRIBUTION ===")

print(
    pd.crosstab(
        df["timeToEntry"],
        df["split"]
    )
)


print("\n=== ENTRY DISTRIBUTION ===")

print(
    pd.crosstab(
        df["entryId"],
        df["split"]
    )
)


# ----------------------------------------------------------
# save
# ----------------------------------------------------------

for name in ["train", "val", "test"]:

    out = (
        OUT_DIR /
        f"{name}_samples.csv"
    )

    df[
        df["split"] == name
    ].reset_index(
        drop=True
    ).to_csv(
        out,
        index=False
    )

    print(f"Saved: {out}")


all_out = (
    ROOT /
    "data/processed/all_decision_samples.csv"
)

df.to_csv(
    all_out,
    index=False
)

print(f"Saved: {all_out}")

print("=" * 100)
