from pathlib import Path
import sys
import time
import random
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.trajectory_dataset import RoundaboutTrajectoryDataset
from models.transformer_trajectory import TransformerTrajectoryModel


SEED = 42
BATCH_SIZE = 64
EPOCHS = 15
PATIENCE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
TRAJ_SCALE = 30.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT = ROOT / "outputs/transformer_trajectory"
OUT.mkdir(parents=True, exist_ok=True)

BEST = OUT / "best.pt"
HISTORY = OUT / "history.csv"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def make_loader(split, shuffle):
    ds = RoundaboutTrajectoryDataset(
        split=split,
        future_sec=4.0,
        future_step_sec=0.2,
    )

    print(f"{split}: {len(ds):,}")

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


print("=" * 100)
print("TRANSFORMER FUTURE TRAJECTORY TRAINING")
print("=" * 100)
print("Device:", DEVICE)

train_loader = make_loader("train", True)
val_loader = make_loader("val", False)
test_loader = make_loader("test", False)


model = TransformerTrajectoryModel(
    future_steps=20
).to(DEVICE)

ckpt_path = ROOT / "outputs/transformer_baseline/best.pt"

ckpt = torch.load(
    ckpt_path,
    map_location=DEVICE,
    weights_only=False
)

load_result = model.load_state_dict(
    ckpt["model"],
    strict=False
)

assert len(load_result.missing_keys) == 6
assert all(
    x.startswith("trajectory_head.")
    for x in load_result.missing_keys
)
assert not load_result.unexpected_keys

print("Loaded:", ckpt_path)
print("Total parameters:",
      sum(p.numel() for p in model.parameters()))


# Freeze all pretrained Transformer parameters.
for p in model.parameters():
    p.requires_grad = False

# Train only the new trajectory head.
for p in model.trajectory_head.parameters():
    p.requires_grad = True

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)

optimizer = torch.optim.AdamW(
    model.trajectory_head.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


def move(batch):
    keys = [
        "agents",
        "agent_mask",
        "entry_line",
        "future_trajectory",
        "future_mask",
    ]

    return {
        k: batch[k].to(DEVICE, non_blocking=True)
        for k in keys
    }


def trajectory_loss(pred, gt, mask):
    loss = F.smooth_l1_loss(
        pred / TRAJ_SCALE,
        gt / TRAJ_SCALE,
        reduction="none"
    )

    valid = mask.unsqueeze(-1)

    return (
        (loss * valid).sum()
        / (valid.sum() * 2.0).clamp_min(1.0)
    )


@torch.no_grad()
def evaluate(loader):
    model.eval()

    ade_sum = 0.0
    ade_count = 0

    fde_sum = 0.0
    fde_count = 0

    loss_sum = 0.0
    count = 0

    for batch in loader:
        b = move(batch)

        out = model(
            b["agents"],
            b["agent_mask"],
            b["entry_line"]
        )

        pred = out["future_trajectory"]
        gt = b["future_trajectory"]
        mask = b["future_mask"]

        loss = trajectory_loss(pred, gt, mask)

        distance = torch.linalg.vector_norm(
            pred - gt,
            dim=-1
        )

        valid = mask > 0

        ade_sum += distance[valid].sum().item()
        ade_count += valid.sum().item()

        # Last valid future waypoint per sample.
        indices = torch.arange(
            distance.shape[1],
            device=DEVICE
        ).unsqueeze(0).expand_as(distance)

        last = torch.where(
            valid,
            indices,
            -1
        ).max(dim=1).values

        has_future = last >= 0

        rows = torch.arange(
            len(last),
            device=DEVICE
        )[has_future]

        fde_sum += distance[
            rows,
            last[has_future]
        ].sum().item()

        fde_count += has_future.sum().item()

        n = len(pred)
        loss_sum += loss.item() * n
        count += n

    return {
        "loss": loss_sum / count,
        "ade": ade_sum / ade_count,
        "fde": fde_sum / fde_count,
    }


best_ade = float("inf")
bad_epochs = 0

with open(HISTORY, "w", newline="") as f:
    csv.writer(f).writerow([
        "epoch", "train_loss", "val_loss",
        "val_ade", "val_fde", "epoch_sec"
    ])


for epoch in range(1, EPOCHS + 1):
    start = time.time()

    # Keep the frozen Transformer in evaluation mode.
    model.eval()
    model.trajectory_head.train()

    running = 0.0
    seen = 0

    for batch in train_loader:
        b = move(batch)

        optimizer.zero_grad(set_to_none=True)

        out = model(
            b["agents"],
            b["agent_mask"],
            b["entry_line"]
        )

        loss = trajectory_loss(
            out["future_trajectory"],
            b["future_trajectory"],
            b["future_mask"]
        )

        loss.backward()
        optimizer.step()

        n = len(b["agents"])
        running += loss.item() * n
        seen += n

    val = evaluate(val_loader)
    sec = time.time() - start

    print(
        f"Epoch {epoch:02d} | "
        f"{sec:.1f}s | "
        f"train={running/seen:.5f} | "
        f"val={val['loss']:.5f} | "
        f"ADE={val['ade']:.4f}m | "
        f"FDE={val['fde']:.4f}m",
        flush=True
    )

    with open(HISTORY, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch, running / seen, val["loss"],
            val["ade"], val["fde"], sec
        ])

    if val["ade"] < best_ade:
        best_ade = val["ade"]
        bad_epochs = 0

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "val_stats": val,
        }, BEST)
    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print("Early stopping.")
        break


best = torch.load(
    BEST,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(best["model"])

test = evaluate(test_loader)

print()
print("=" * 100)
print("TEST TRANSFORMER TRAJECTORY RESULTS")
print("=" * 100)
print("Best epoch :", best["epoch"])
print(f"ADE        : {test['ade']:.4f} m")
print(f"FDE        : {test['fde']:.4f} m")
print("Checkpoint :", BEST)
print("History    :", HISTORY)
print("=" * 100)
