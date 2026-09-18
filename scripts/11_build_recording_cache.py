from pathlib import Path
import numpy as np
import pandas as pd
import time
import os

ROOT = Path.home() / "roundabout_ssm"
DATA = ROOT / "data/raw/data"
CACHE_DIR = ROOT / "data/cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RECORDINGS = range(2, 24)

STATE_COLS = [
    "xCenter",
    "yCenter",
    "heading",
    "xVelocity",
    "yVelocity",
    "xAcceleration",
    "yAcceleration",
    "lonVelocity",
    "latVelocity",
    "lonAcceleration",
    "latAcceleration",
]

CLASS_TO_ID = {
    "car": 0,
    "van": 1,
    "truck": 2,
    "bus": 3,
    "trailer": 4,
    "motorcycle": 5,
    "bicycle": 6,
    "pedestrian": 7,
}

print("=" * 100)
print("BUILD RECORDING CACHE")
print("=" * 100)

total_rows = 0
total_size = 0

for rec in RECORDINGS:

    rid = f"{rec:02d}"
    start = time.time()

    print(f"\n[{rid}] loading CSV ...")

    tracks = pd.read_csv(
        DATA / f"{rid}_tracks.csv",
        usecols=["trackId", "frame"] + STATE_COLS
    )

    meta = pd.read_csv(
        DATA / f"{rid}_tracksMeta.csv"
    )

    # ------------------------------------------------------
    # Sort by track -> frame
    # This makes each vehicle trajectory contiguous in memory.
    # ------------------------------------------------------
    tracks = tracks.sort_values(
        ["trackId", "frame"]
    ).reset_index(drop=True)

    track_ids = tracks["trackId"].to_numpy(
        dtype=np.int32
    )

    frames = tracks["frame"].to_numpy(
        dtype=np.int32
    )

    states = tracks[STATE_COLS].to_numpy(
        dtype=np.float32
    )

    n_rows = len(tracks)

    # ------------------------------------------------------
    # Track index
    #
    # unique_track_ids[i] trajectory:
    # rows track_ptr[i] : track_ptr[i+1]
    # ------------------------------------------------------
    unique_track_ids, track_first = np.unique(
        track_ids,
        return_index=True
    )

    track_ptr = np.concatenate([
        track_first.astype(np.int64),
        np.array([n_rows], dtype=np.int64)
    ])

    class_map = dict(
        zip(
            meta["trackId"].astype(int),
            meta["class"].map(CLASS_TO_ID).fillna(-1).astype(int)
        )
    )

    width_map = dict(
        zip(
            meta["trackId"].astype(int),
            meta["width"].astype(float)
        )
    )

    length_map = dict(
        zip(
            meta["trackId"].astype(int),
            meta["length"].astype(float)
        )
    )

    track_class = np.array(
        [
            class_map.get(int(t), -1)
            for t in unique_track_ids
        ],
        dtype=np.int8
    )

    track_width = np.array(
        [
            width_map.get(int(t), np.nan)
            for t in unique_track_ids
        ],
        dtype=np.float32
    )

    track_length = np.array(
        [
            length_map.get(int(t), np.nan)
            for t in unique_track_ids
        ],
        dtype=np.float32
    )

    # ------------------------------------------------------
    # Frame index
    #
    # For a current frame we need all agents that exist
    # at exactly that frame for neighbor selection.
    #
    # frame_row_indices stores references back into
    # track-sorted states[].
    # ------------------------------------------------------
    frame_order = np.argsort(
        frames,
        kind="stable"
    ).astype(np.int64)

    sorted_frames = frames[frame_order]

    unique_frames, frame_first = np.unique(
        sorted_frames,
        return_index=True
    )

    frame_ptr = np.concatenate([
        frame_first.astype(np.int64),
        np.array([n_rows], dtype=np.int64)
    ])

    # ------------------------------------------------------
    # Recording metadata
    # ------------------------------------------------------
    recording_meta = pd.read_csv(
        DATA / f"{rid}_recordingMeta.csv"
    ).iloc[0]

    frame_rate = np.float32(
        recording_meta["frameRate"]
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------
    out = CACHE_DIR / f"{rid}.npz"

    np.savez_compressed(
        out,

        # row-level arrays
        track_ids=track_ids,
        frames=frames,
        states=states,

        # track index
        unique_track_ids=unique_track_ids.astype(np.int32),
        track_ptr=track_ptr,
        track_class=track_class,
        track_width=track_width,
        track_length=track_length,

        # frame index
        unique_frames=unique_frames.astype(np.int32),
        frame_ptr=frame_ptr,
        frame_row_indices=frame_order,

        # metadata
        frame_rate=frame_rate,

        state_columns=np.array(
            STATE_COLS,
            dtype="U32"
        ),

        class_names=np.array(
            [
                "car",
                "van",
                "truck",
                "bus",
                "trailer",
                "motorcycle",
                "bicycle",
                "pedestrian",
            ],
            dtype="U16"
        )
    )

    size_mb = os.path.getsize(out) / 1024 / 1024
    elapsed = time.time() - start

    total_rows += n_rows
    total_size += size_mb

    print(
        f"[{rid}] "
        f"rows={n_rows:,} | "
        f"tracks={len(unique_track_ids):,} | "
        f"frames={len(unique_frames):,} | "
        f"{size_mb:.1f} MB | "
        f"{elapsed:.1f}s"
    )

print()
print("=" * 100)
print("CACHE COMPLETE")
print("=" * 100)
print(f"Total rows : {total_rows:,}")
print(f"Total cache: {total_size:.1f} MB")
print(f"Directory  : {CACHE_DIR}")
print("=" * 100)
