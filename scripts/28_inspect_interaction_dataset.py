from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.interaction_dataset import (
    RoundaboutInteractionDataset
)


print("=" * 100)
print("INTERACTION DATASET TEST")
print("=" * 100)

ds = RoundaboutInteractionDataset(
    split="train",
    history_frames=50,
    max_neighbors=8,
    neighbor_radius=30.0,
    max_cached_recordings=32,
)

print("Dataset length:", len(ds))

indices = [
    0,
    len(ds) // 4,
    len(ds) // 2,
    len(ds) - 1,
]

names = ds.FEATURE_COLS


for idx in indices:

    s = ds[idx]

    print()
    print("-" * 100)

    print("INDEX:", idx)

    print(
        "Rec / Track / Frame / Entry:",
        int(s["recording_id"]),
        int(s["track_id"]),
        int(s["current_frame"]),
        int(s["entry_id"]),
    )

    print(
        "agents:",
        tuple(s["agents"].shape)
    )

    print(
        "interaction:",
        tuple(s["interaction"].shape)
    )

    print()

    raw = (
        s["interaction_raw"]
        .numpy()
    )

    norm = (
        s["interaction"]
        .numpy()
    )

    for name, r, n in zip(
        names,
        raw,
        norm
    ):

        print(
            f"{name:23s} "
            f"raw={r:9.4f} | "
            f"norm={n:9.4f}"
        )

    print()

    print(
        "decision:",
        int(s["decision"])
    )

    print(
        "TTE:",
        float(s["time_to_entry"])
    )


# ----------------------------------------------------------
# Numerical QA on 10,000 samples
# ----------------------------------------------------------

N = min(
    10000,
    len(ds)
)

values = []

for i in range(N):

    values.append(
        ds[i]["interaction"].numpy()
    )

values = np.stack(
    values
)

print()
print("=" * 100)
print(f"NORMALIZATION QA - FIRST {N} TRAIN SAMPLES")
print("=" * 100)

print(
    "All finite:",
    np.isfinite(values).all()
)

print()
print("Feature mean:")
print(
    np.round(
        values.mean(axis=0),
        3
    )
)

print()
print("Feature std:")
print(
    np.round(
        values.std(axis=0),
        3
    )
)

print()
print("=" * 100)
