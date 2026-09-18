from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home() / "roundabout_ssm"

EVENTS = ROOT / "data/processed/location0_entry_crossings.csv"
DATA = ROOT / "data/raw/data"

HISTORY_SEC = 2.0

# 실제 진입 전 몇 초 지점을 decision sample 후보로 사용할지
OFFSETS_SEC = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0
]

# GO/WAIT threshold 후보
THRESHOLDS = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0
]

events = pd.read_csv(EVENTS)

samples = []

print("=" * 100)
print("DECISION LABEL ANALYSIS")
print("=" * 100)
print(f"History length: {HISTORY_SEC:.1f} sec")
print()

for rec_id, rec_events in events.groupby("recordingId"):

    rid = f"{int(rec_id):02d}"

    meta = pd.read_csv(
        DATA / f"{rid}_tracksMeta.csv"
    )

    initial_frame = dict(
        zip(
            meta["trackId"].astype(int),
            meta["initialFrame"].astype(int)
        )
    )

    for _, event in rec_events.iterrows():

        track_id = int(event["trackId"])
        cross_frame = int(event["crossFrame"])
        frame_rate = float(event["frameRate"])

        if track_id not in initial_frame:
            continue

        first_frame = initial_frame[track_id]

        for tte in OFFSETS_SEC:

            current_frame = (
                cross_frame -
                int(round(tte * frame_rate))
            )

            history_start = (
                current_frame -
                int(round(HISTORY_SEC * frame_rate))
            )

            # 과거 history가 충분한 sample만 사용
            if history_start < first_frame:
                continue

            samples.append({
                "recordingId": int(rec_id),
                "trackId": track_id,
                "class": event["class"],
                "entryId": int(event["entryId"]),
                "currentFrame": current_frame,
                "crossFrame": cross_frame,
                "timeToEntry": tte
            })

df = pd.DataFrame(samples)

print(f"Candidate samples: {len(df)}")
print(f"Unique vehicles : {df[['recordingId','trackId']].drop_duplicates().shape[0]}")

print("\n=== SAMPLES BY TIME-TO-ENTRY ===")

tte_table = (
    df["timeToEntry"]
    .value_counts()
    .sort_index()
)

print(tte_table)


print("\n=== GO / WAIT THRESHOLD TEST ===")

rows = []

for threshold in THRESHOLDS:

    go = int(
        (df["timeToEntry"] <= threshold).sum()
    )

    wait = int(
        (df["timeToEntry"] > threshold).sum()
    )

    total = go + wait

    go_pct = 100 * go / total
    wait_pct = 100 * wait / total

    ratio = (
        min(go, wait) /
        max(go, wait)
        if max(go, wait) > 0
        else 0
    )

    rows.append({
        "threshold_sec": threshold,
        "GO": go,
        "WAIT": wait,
        "GO_pct": go_pct,
        "WAIT_pct": wait_pct,
        "minority_majority_ratio": ratio
    })

result = pd.DataFrame(rows)

print(
    result.to_string(
        index=False,
        formatters={
            "GO_pct": "{:.2f}".format,
            "WAIT_pct": "{:.2f}".format,
            "minority_majority_ratio": "{:.3f}".format
        }
    )
)


print("\n=== ENTRY DISTRIBUTION ===")

print(
    pd.crosstab(
        df["entryId"],
        df["timeToEntry"]
    )
)


print("\n=== UNIQUE VEHICLES BY MAX AVAILABLE OFFSET ===")

vehicle_max = (
    df.groupby(["recordingId", "trackId"])
    ["timeToEntry"]
    .max()
)

print(
    vehicle_max
    .value_counts()
    .sort_index()
)


OUT = (
    ROOT /
    "data/processed/decision_sample_candidates.csv"
)

df.to_csv(
    OUT,
    index=False
)

print()
print(f"Saved: {OUT}")
print("=" * 100)
