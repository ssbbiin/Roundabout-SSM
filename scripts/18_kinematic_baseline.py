from pathlib import Path
import sys
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.torch_dataset import RoundaboutTorchDataset


BATCH_SIZE = 512
NUM_WORKERS = 2

GO_THRESHOLD = 1.6

# 너무 느린 차량에서 distance/speed가 폭발하는 것 방지
MIN_SPEED = 0.3
MAX_TTE = 10.0


print("=" * 90)
print("KINEMATIC CONSTANT-VELOCITY BASELINE")
print("=" * 90)


ds = RoundaboutTorchDataset(
    split="test",
    history_frames=50,
    max_neighbors=8,
    neighbor_radius=30.0,
    max_cached_recordings=32,
)

loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)


total = 0

tp = 0
tn = 0
fp = 0
fn = 0

tte_abs = 0.0
speed_abs = 0.0
heading_abs = 0.0

distance_sum = 0.0
current_speed_sum = 0.0


for batch in loader:

    agents = batch["agents"]

    # ------------------------------------------------------
    # Current Ego
    #
    # feature:
    # 0 rel_x
    # 1 rel_y
    # 2 rel_vx
    # 3 rel_vy
    # ...
    # ------------------------------------------------------

    ego = agents[:, 0, -1, :]

    vx = ego[:, 2]
    vy = ego[:, 3]

    current_speed = torch.sqrt(
        vx ** 2 + vy ** 2
    )

    # ------------------------------------------------------
    # Entry-line midpoint in Ego coordinates
    # ------------------------------------------------------

    entry_line = batch["entry_line"]

    midpoint = entry_line.mean(dim=1)

    entry_x = midpoint[:, 0]
    entry_y = midpoint[:, 1]

    distance = torch.sqrt(
        entry_x ** 2 +
        entry_y ** 2
    )

    # ------------------------------------------------------
    # Constant velocity TTE
    # ------------------------------------------------------

    safe_speed = torch.clamp(
        current_speed,
        min=MIN_SPEED
    )

    pred_tte = distance / safe_speed

    pred_tte = torch.clamp(
        pred_tte,
        max=MAX_TTE
    )

    pred_decision = (
        pred_tte <= GO_THRESHOLD
    ).long()

    gt_decision = batch["decision"]

    # ------------------------------------------------------
    # Entry-speed baseline:
    # future entry speed = current speed
    # ------------------------------------------------------

    pred_entry_speed = current_speed

    # ------------------------------------------------------
    # Heading baseline:
    # assume no heading change.
    #
    # Relative heading:
    # sin = 0
    # cos = 1
    # ------------------------------------------------------

    pred_heading = torch.zeros(
        (
            len(agents),
            2
        ),
        dtype=torch.float32
    )

    pred_heading[:, 1] = 1.0

    gt_heading = batch[
        "entry_heading"
    ]

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    n = len(agents)
    total += n

    tp += (
        (pred_decision == 1) &
        (gt_decision == 1)
    ).sum().item()

    tn += (
        (pred_decision == 0) &
        (gt_decision == 0)
    ).sum().item()

    fp += (
        (pred_decision == 1) &
        (gt_decision == 0)
    ).sum().item()

    fn += (
        (pred_decision == 0) &
        (gt_decision == 1)
    ).sum().item()

    tte_abs += torch.abs(
        pred_tte -
        batch["time_to_entry"]
    ).sum().item()

    speed_abs += torch.abs(
        pred_entry_speed -
        batch["entry_speed"]
    ).sum().item()

    # heading error
    gt_angle = torch.atan2(
        gt_heading[:, 0],
        gt_heading[:, 1]
    )

    pred_angle = torch.atan2(
        pred_heading[:, 0],
        pred_heading[:, 1]
    )

    diff = torch.atan2(
        torch.sin(
            pred_angle - gt_angle
        ),
        torch.cos(
            pred_angle - gt_angle
        )
    )

    heading_abs += (
        torch.abs(diff)
        * 180.0
        / math.pi
    ).sum().item()

    distance_sum += (
        distance.sum().item()
    )

    current_speed_sum += (
        current_speed.sum().item()
    )


accuracy = (
    (tp + tn) / total
)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0.0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0.0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0.0
)


print()
print("Samples          :", total)

print()
print("Accuracy         :", f"{accuracy*100:.2f}%")
print("Precision        :", f"{precision:.4f}")
print("Recall           :", f"{recall:.4f}")
print("F1               :", f"{f1:.4f}")

print()
print(
    "TTE MAE          :",
    f"{tte_abs/total:.4f} s"
)

print(
    "Entry speed MAE  :",
    f"{speed_abs/total:.4f} m/s"
)

print(
    "Heading MAE      :",
    f"{heading_abs/total:.3f} deg"
)

print()
print(
    "Mean distance    :",
    f"{distance_sum/total:.3f} m"
)

print(
    "Mean current spd :",
    f"{current_speed_sum/total:.3f} m/s"
)

print()
print("Confusion matrix:")
print()
print("              Pred WAIT   Pred GO")

print(
    f"True WAIT     "
    f"{tn:9d} "
    f"{fp:9d}"
)

print(
    f"True GO       "
    f"{fn:9d} "
    f"{tp:9d}"
)

print()
print("=" * 90)
