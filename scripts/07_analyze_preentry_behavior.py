from pathlib import Path
import pandas as pd
import numpy as np
import math

ROOT = Path.home() / "roundabout_ssm"
DATA = ROOT / "data/raw/data"

EVENTS_FILE = ROOT / "data/processed/location0_entry_crossings.csv"
OUT_FILE = ROOT / "data/processed/location0_preentry_stats.csv"

WINDOW_SEC = 5.0

STOP_THRESHOLD = 0.5      # m/s
SLOW_THRESHOLD = 1.0      # m/s


def longest_consecutive_duration(mask, frame_rate):
    max_run = 0
    run = 0

    for v in mask:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    return max_run / frame_rate


def value_at_seconds_before(group, cross_frame, sec, frame_rate, column):
    target = cross_frame - int(round(sec * frame_rate))

    rows = group[group["frame"] == target]

    if len(rows) == 0:
        return np.nan

    return float(rows.iloc[0][column])


events = pd.read_csv(EVENTS_FILE)

all_stats = []

print("=" * 100)
print("PRE-ENTRY BEHAVIOR ANALYSIS")
print("=" * 100)

for rec_id, rec_events in events.groupby("recordingId"):

    rid = f"{int(rec_id):02d}"

    print(f"Processing recording {rid} ...")

    tracks = pd.read_csv(
        DATA / f"{rid}_tracks.csv",
        usecols=[
            "trackId",
            "frame",
            "xCenter",
            "yCenter",
            "xVelocity",
            "yVelocity",
            "xAcceleration",
            "yAcceleration",
            "lonVelocity",
            "latVelocity",
            "lonAcceleration",
            "latAcceleration",
            "heading"
        ]
    )

    tracks["speed"] = np.sqrt(
        tracks["xVelocity"] ** 2 +
        tracks["yVelocity"] ** 2
    )

    track_groups = {
        int(track_id): g.sort_values("frame").reset_index(drop=True)
        for track_id, g in tracks.groupby("trackId")
    }

    for _, event in rec_events.iterrows():

        track_id = int(event["trackId"])
        cross_frame = int(event["crossFrame"])
        frame_rate = float(event["frameRate"])

        if track_id not in track_groups:
            continue

        g = track_groups[track_id]

        first_frame = int(g["frame"].min())

        history_available_sec = (
            cross_frame - first_frame
        ) / frame_rate

        window_frames = int(round(WINDOW_SEC * frame_rate))

        history = g[
            (g["frame"] >= cross_frame - window_frames) &
            (g["frame"] <= cross_frame)
        ].copy()

        # crossing frame 이전만 행동 통계에 사용
        pre = history[
            history["frame"] < cross_frame
        ].copy()

        if len(pre) == 0:
            continue

        speed = pre["speed"].to_numpy()

        stop_mask = speed < STOP_THRESHOLD
        slow_mask = speed < SLOW_THRESHOLD

        stop_duration_total = (
            stop_mask.sum() / frame_rate
        )

        slow_duration_total = (
            slow_mask.sum() / frame_rate
        )

        longest_stop = longest_consecutive_duration(
            stop_mask,
            frame_rate
        )

        longest_slow = longest_consecutive_duration(
            slow_mask,
            frame_rate
        )

        row = {
            "recordingId": int(rec_id),
            "trackId": track_id,
            "class": event["class"],
            "entryId": int(event["entryId"]),
            "crossFrame": cross_frame,

            "historyAvailableSec": history_available_sec,

            "has1sHistory": history_available_sec >= 1.0,
            "has2sHistory": history_available_sec >= 2.0,
            "has3sHistory": history_available_sec >= 3.0,
            "has4sHistory": history_available_sec >= 4.0,
            "has5sHistory": history_available_sec >= 5.0,

            "crossSpeed": float(event["speed"]),

            "speedMinus1s": value_at_seconds_before(
                g, cross_frame, 1, frame_rate, "speed"
            ),

            "speedMinus2s": value_at_seconds_before(
                g, cross_frame, 2, frame_rate, "speed"
            ),

            "speedMinus3s": value_at_seconds_before(
                g, cross_frame, 3, frame_rate, "speed"
            ),

            "speedMinus4s": value_at_seconds_before(
                g, cross_frame, 4, frame_rate, "speed"
            ),

            "speedMinus5s": value_at_seconds_before(
                g, cross_frame, 5, frame_rate, "speed"
            ),

            "minSpeedLast5s": float(pre["speed"].min()),
            "meanSpeedLast5s": float(pre["speed"].mean()),
            "maxSpeedLast5s": float(pre["speed"].max()),

            "minLonAccelerationLast5s":
                float(pre["lonAcceleration"].min()),

            "meanLonAccelerationLast5s":
                float(pre["lonAcceleration"].mean()),

            "stopDurationLast5s":
                stop_duration_total,

            "slowDurationLast5s":
                slow_duration_total,

            "longestStopLast5s":
                longest_stop,

            "longestSlowLast5s":
                longest_slow,

            "stoppedAtLeast0_5s":
                longest_stop >= 0.5,

            "stoppedAtLeast1s":
                longest_stop >= 1.0
        }

        all_stats.append(row)


df = pd.DataFrame(all_stats)

df.to_csv(
    OUT_FILE,
    index=False
)

print()
print("=" * 100)
print("RESULT")
print("=" * 100)

print(f"Vehicles analyzed: {len(df)}")

print("\n=== HISTORY AVAILABILITY ===")

for sec in [1, 2, 3, 4, 5]:
    col = f"has{sec}sHistory"
    n = int(df[col].sum())
    pct = 100 * n / len(df)

    print(
        f">= {sec}s history: "
        f"{n:5d} / {len(df)} "
        f"({pct:6.2f}%)"
    )


print("\n=== SPEED BEFORE ENTRY [m/s] ===")

speed_cols = [
    "speedMinus5s",
    "speedMinus4s",
    "speedMinus3s",
    "speedMinus2s",
    "speedMinus1s",
    "crossSpeed"
]

print(
    df[speed_cols]
    .describe()
    .T
    .to_string()
)


print("\n=== STOP / SLOW BEHAVIOR ===")

stop05 = int(df["stoppedAtLeast0_5s"].sum())
stop10 = int(df["stoppedAtLeast1s"].sum())

print(
    f"Stopped (< {STOP_THRESHOLD} m/s) "
    f"for >=0.5 s: "
    f"{stop05} "
    f"({100*stop05/len(df):.2f}%)"
)

print(
    f"Stopped (< {STOP_THRESHOLD} m/s) "
    f"for >=1.0 s: "
    f"{stop10} "
    f"({100*stop10/len(df):.2f}%)"
)


print("\nLongest stop duration [s]:")

print(
    df["longestStopLast5s"]
    .describe(
        percentiles=[
            .10, .25, .50, .75, .90, .95, .99
        ]
    )
)


print("\n=== BY ENTRY ===")

entry_stats = (
    df.groupby("entryId")
    .agg(
        vehicles=("trackId", "count"),
        mean_cross_speed=("crossSpeed", "mean"),
        median_cross_speed=("crossSpeed", "median"),
        mean_speed_minus3s=("speedMinus3s", "mean"),
        stop_05_rate=("stoppedAtLeast0_5s", "mean"),
        stop_10_rate=("stoppedAtLeast1s", "mean"),
    )
)

entry_stats["stop_05_rate"] *= 100
entry_stats["stop_10_rate"] *= 100

print(entry_stats.to_string())


print("\n=== BY CLASS ===")

class_stats = (
    df.groupby("class")
    .agg(
        vehicles=("trackId", "count"),
        mean_cross_speed=("crossSpeed", "mean"),
        stop_rate=("stoppedAtLeast0_5s", "mean")
    )
)

class_stats["stop_rate"] *= 100

print(class_stats.to_string())


print()
print(f"Saved: {OUT_FILE}")
print("=" * 100)
