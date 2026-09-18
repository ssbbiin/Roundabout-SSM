from pathlib import Path
import json
import math
import pandas as pd
import numpy as np

ROOT = Path.home() / "roundabout_ssm"
DATA = ROOT / "data/raw/data"
GEOM = ROOT / "configs/location0_geometry.json"

OUT_DIR = ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "location0_entry_crossings.csv"

MOTOR_CLASSES = {
    "car",
    "van",
    "truck",
    "bus",
    "trailer"
}


# ----------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------

def orientation(a, b, c):
    """
    Signed orientation of three 2D points.
    """
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def segments_intersect(a, b, c, d):
    """
    True if segment AB intersects segment CD.
    """
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    eps = 1e-9

    return (
        ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps))
        and
        ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))
    )


def dist(p, q):
    return math.hypot(
        p[0] - q[0],
        p[1] - q[1]
    )


# ----------------------------------------------------------
# load geometry
# ----------------------------------------------------------

with open(GEOM, "r") as f:
    geom = json.load(f)

entries = []

for e in geom["entries"]:

    p1 = tuple(e["p1_world"])
    p2 = tuple(e["p2_world"])

    midpoint = (
        (p1[0] + p2[0]) / 2.0,
        (p1[1] + p2[1]) / 2.0
    )

    entries.append({
        "entryId": int(e["id"]),
        "p1": p1,
        "p2": p2,
        "midpoint": midpoint
    })


# Roundabout center:
# average of four entry-line midpoints
center_x = np.mean([e["midpoint"][0] for e in entries])
center_y = np.mean([e["midpoint"][1] for e in entries])

CENTER = (center_x, center_y)

print("=" * 90)
print("ENTRY CROSSING EXTRACTION")
print("=" * 90)
print(f"Estimated roundabout center: ({center_x:.3f}, {center_y:.3f})")
print()


events = []


# ----------------------------------------------------------
# process recordings 02 ~ 23
# ----------------------------------------------------------

for rec in range(2, 24):

    rid = f"{rec:02d}"

    print(f"Processing recording {rid} ...")

    recording_meta = pd.read_csv(
        DATA / f"{rid}_recordingMeta.csv"
    ).iloc[0]

    frame_rate = float(recording_meta["frameRate"])

    track_meta = pd.read_csv(
        DATA / f"{rid}_tracksMeta.csv"
    )

    tracks = pd.read_csv(
        DATA / f"{rid}_tracks.csv"
    )

    # Keep motor vehicles only
    valid_meta = track_meta[
        track_meta["class"].isin(MOTOR_CLASSES)
    ].copy()

    class_map = dict(
        zip(
            valid_meta["trackId"],
            valid_meta["class"]
        )
    )

    valid_ids = set(valid_meta["trackId"].tolist())

    tracks = tracks[
        tracks["trackId"].isin(valid_ids)
    ].copy()

    for track_id, group in tracks.groupby("trackId"):

        group = group.sort_values("frame").reset_index(drop=True)

        if len(group) < 2:
            continue

        # A vehicle should enter the roundabout only once.
        found_event = None

        for i in range(len(group) - 1):

            r0 = group.iloc[i]
            r1 = group.iloc[i + 1]

            a = (
                float(r0["xCenter"]),
                float(r0["yCenter"])
            )

            b = (
                float(r1["xCenter"]),
                float(r1["yCenter"])
            )

            for entry in entries:

                c = entry["p1"]
                d = entry["p2"]

                if not segments_intersect(a, b, c, d):
                    continue

                # --------------------------------------------------
                # Direction check
                #
                # Entry crossing should move toward roundabout center.
                # --------------------------------------------------

                before_radius = dist(a, CENTER)
                after_radius = dist(b, CENTER)

                if after_radius >= before_radius:
                    continue

                speed = math.hypot(
                    float(r1["xVelocity"]),
                    float(r1["yVelocity"])
                )

                acceleration = math.hypot(
                    float(r1["xAcceleration"]),
                    float(r1["yAcceleration"])
                )

                found_event = {
                    "recordingId": rec,
                    "trackId": int(track_id),
                    "class": class_map[int(track_id)],
                    "entryId": entry["entryId"],

                    "crossFrame": int(r1["frame"]),
                    "crossTimeSec": float(r1["frame"]) / frame_rate,

                    "x": float(r1["xCenter"]),
                    "y": float(r1["yCenter"]),

                    "heading": float(r1["heading"]),

                    "xVelocity": float(r1["xVelocity"]),
                    "yVelocity": float(r1["yVelocity"]),
                    "speed": speed,

                    "xAcceleration": float(r1["xAcceleration"]),
                    "yAcceleration": float(r1["yAcceleration"]),
                    "acceleration": acceleration,

                    "lonVelocity": float(r1["lonVelocity"]),
                    "latVelocity": float(r1["latVelocity"]),

                    "lonAcceleration": float(r1["lonAcceleration"]),
                    "latAcceleration": float(r1["latAcceleration"]),

                    "frameRate": frame_rate
                }

                break

            if found_event is not None:
                break

        if found_event is not None:
            events.append(found_event)


# ----------------------------------------------------------
# save
# ----------------------------------------------------------

events_df = pd.DataFrame(events)

events_df.to_csv(
    OUT_FILE,
    index=False
)

print()
print("=" * 90)
print("RESULT")
print("=" * 90)

print(f"Total crossing events: {len(events_df)}")

if len(events_df) > 0:

    print("\n=== BY ENTRY ===")
    print(
        events_df["entryId"]
        .value_counts()
        .sort_index()
    )

    print("\n=== BY CLASS ===")
    print(
        events_df["class"]
        .value_counts()
    )

    print("\n=== ENTRY x CLASS ===")
    print(
        pd.crosstab(
            events_df["entryId"],
            events_df["class"]
        )
    )

    print("\n=== ENTRY SPEED [m/s] ===")
    print(
        events_df["speed"]
        .describe()
    )

    print("\n=== FIRST 20 EVENTS ===")
    print(
        events_df[
            [
                "recordingId",
                "trackId",
                "class",
                "entryId",
                "crossFrame",
                "crossTimeSec",
                "speed",
                "heading"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

print()
print(f"Saved: {OUT_FILE}")
print("=" * 90)
