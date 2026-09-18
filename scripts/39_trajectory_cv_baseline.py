from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.trajectory_dataset import (
    RoundaboutTrajectoryDataset
)


BATCH_SIZE = 512
NUM_WORKERS = 2

FUTURE_STEP = 0.2
FUTURE_STEPS = 20


print("=" * 100)
print("CONSTANT-VELOCITY FUTURE TRAJECTORY BASELINE")
print("=" * 100)


ds = RoundaboutTrajectoryDataset(
    split="test",
    future_sec=4.0,
    future_step_sec=0.2,
)

loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)


ade_sum = 0.0
ade_count = 0

fde_sum = 0.0
fde_count = 0


for batch in loader:

    # Current Ego feature in Ego-centric coordinates
    ego = batch[
        "agents"
    ][:, 0, -1, :]

    vx = ego[:, 2]
    vy = ego[:, 3]

    B = len(ego)

    times = (
        torch.arange(
            1,
            FUTURE_STEPS + 1,
            dtype=torch.float32
        )
        * FUTURE_STEP
    )

    # [B,T]
    pred_x = (
        vx[:, None]
        * times[None, :]
    )

    pred_y = (
        vy[:, None]
        * times[None, :]
    )

    # [B,T,2]
    pred = torch.stack(
        [
            pred_x,
            pred_y
        ],
        dim=-1
    )

    gt = batch[
        "future_trajectory"
    ]

    mask = (
        batch[
            "future_mask"
        ] > 0
    )

    distance = torch.sqrt(
        (
            (pred - gt) ** 2
        ).sum(dim=-1)
    )

    # ADE
    ade_sum += (
        distance[mask]
        .sum()
        .item()
    )

    ade_count += int(
        mask.sum()
        .item()
    )

    # FDE = last valid waypoint
    for i in range(B):

        idx = torch.where(
            mask[i]
        )[0]

        if len(idx) == 0:
            continue

        last = int(
            idx[-1]
        )

        fde_sum += float(
            distance[
                i,
                last
            ]
        )

        fde_count += 1


ade = (
    ade_sum
    / ade_count
)

fde = (
    fde_sum
    / fde_count
)


print(
    f"Samples         : {len(ds):,}"
)

print(
    f"Trajectory ADE  : {ade:.4f} m"
)

print(
    f"Trajectory FDE  : {fde:.4f} m"
)

print()
print("=" * 100)

print(
    "Mamba target comparison:"
)

print(
    "Mamba ADE      : 0.7742 m"
)

print(
    "Mamba FDE      : 1.8905 m"
)

print("=" * 100)
