from pathlib import Path
import sys
import time
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.interaction_dataset import RoundaboutInteractionDataset
from models.mamba_residual_interaction import MambaResidualInteraction


SEED = 42

BATCH_SIZE = 64
NUM_WORKERS = 2

EPOCHS = 10
PATIENCE = 4

BASE_LR = 2e-5
INTERACTION_LR = 2e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = ROOT / "outputs/mamba_residual_interaction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST = OUT_DIR / "best.pt"

V1 = ROOT / "outputs/mamba_baseline/best.pt"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 110)
print("MAMBA2 RESIDUAL INTERACTION TRAINING")
print("=" * 110)
print("Device:", DEVICE)


train_ds = RoundaboutInteractionDataset(
    split="train",
    max_cached_recordings=32,
)

val_ds = RoundaboutInteractionDataset(
    split="val",
    max_cached_recordings=32,
)

test_ds = RoundaboutInteractionDataset(
    split="test",
    max_cached_recordings=32,
)

print("Train:", len(train_ds))
print("Val  :", len(val_ds))
print("Test :", len(test_ds))


def loader(ds, shuffle):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


train_loader = loader(train_ds, True)
val_loader = loader(val_ds, False)
test_loader = loader(test_ds, False)


model = MambaResidualInteraction().to(DEVICE)

v1 = torch.load(
    V1,
    map_location=DEVICE,
    weights_only=False
)

result = model.load_state_dict(
    v1["model"],
    strict=False
)

print("Loaded Mamba v1:", V1)
print("Missing new keys:", len(result.missing_keys))
print("Unexpected keys :", len(result.unexpected_keys))
print("Parameters      :", sum(p.numel() for p in model.parameters()))


interaction_params = []
base_params = []

for name, p in model.named_parameters():

    if (
        name.startswith("interaction_encoder")
        or name == "interaction_alpha"
    ):
        interaction_params.append(p)

    else:
        base_params.append(p)


optimizer = torch.optim.AdamW(
    [
        {
            "params": base_params,
            "lr": BASE_LR,
        },
        {
            "params": interaction_params,
            "lr": INTERACTION_LR,
        },
    ],
    weight_decay=1e-4,
)


def move(batch):

    for key in [
        "agents",
        "agent_mask",
        "entry_line",
        "interaction",
        "decision",
        "time_to_entry",
        "entry_speed",
        "entry_heading",
    ]:
        batch[key] = batch[key].to(
            DEVICE,
            non_blocking=True
        )

    return batch


def loss_fn(out, batch):

    cls = F.cross_entropy(
        out["decision_logits"],
        batch["decision"]
    )

    tte = F.smooth_l1_loss(
        out["time_to_entry"] / 4.0,
        batch["time_to_entry"] / 4.0
    )

    speed = F.smooth_l1_loss(
        out["entry_speed"] / 10.0,
        batch["entry_speed"] / 10.0
    )

    heading = F.mse_loss(
        out["entry_heading"],
        batch["entry_heading"]
    )

    return (
        cls
        + tte
        + 0.5 * speed
        + 0.2 * heading
    )


def heading_err(pred, target):

    pred = F.normalize(
        pred,
        dim=1,
        eps=1e-8
    )

    target = F.normalize(
        target,
        dim=1,
        eps=1e-8
    )

    p = torch.atan2(
        pred[:, 0],
        pred[:, 1]
    )

    t = torch.atan2(
        target[:, 0],
        target[:, 1]
    )

    d = torch.atan2(
        torch.sin(p - t),
        torch.cos(p - t)
    )

    return (
        torch.abs(d)
        * 180.0
        / math.pi
    )


@torch.no_grad()
def evaluate(dl):

    model.eval()

    total = 0
    loss_sum = 0.0

    tp = tn = fp = fn = 0

    tte_sum = 0.0
    speed_sum = 0.0
    heading_sum = 0.0

    for batch in dl:

        batch = move(batch)

        out = model(
            batch["agents"],
            batch["agent_mask"],
            batch["entry_line"],
            batch["interaction"],
        )

        loss = loss_fn(
            out,
            batch
        )

        pred = (
            out["decision_logits"]
            .argmax(1)
        )

        gt = batch["decision"]

        n = len(gt)

        total += n
        loss_sum += loss.item() * n

        tp += (
            (pred == 1) & (gt == 1)
        ).sum().item()

        tn += (
            (pred == 0) & (gt == 0)
        ).sum().item()

        fp += (
            (pred == 1) & (gt == 0)
        ).sum().item()

        fn += (
            (pred == 0) & (gt == 1)
        ).sum().item()

        tte_sum += torch.abs(
            out["time_to_entry"]
            - batch["time_to_entry"]
        ).sum().item()

        speed_sum += torch.abs(
            out["entry_speed"]
            - batch["entry_speed"]
        ).sum().item()

        heading_sum += (
            heading_err(
                out["entry_heading"],
                batch["entry_heading"]
            )
            .sum()
            .item()
        )

    acc = (tp + tn) / total

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0
    )

    return {
        "loss": loss_sum / total,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tte_mae": tte_sum / total,
        "speed_mae": speed_sum / total,
        "heading_mae": heading_sum / total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# EPOCH 0: MUST MATCH MAMBA V1
# ============================================================

val0 = evaluate(val_loader)
test0 = evaluate(test_loader)

print()
print("=" * 110)
print("EPOCH 0 — MAMBA V1 INITIALIZATION")
print("=" * 110)

print(
    f"alpha="
    f"{torch.tanh(model.interaction_alpha).item():+.6f}"
)

print(
    f"VAL  | loss={val0['loss']:.4f} | "
    f"F1={val0['f1']:.4f} | "
    f"TTE={val0['tte_mae']:.4f}s"
)

print(
    f"TEST | "
    f"acc={test0['accuracy']*100:.2f}% | "
    f"F1={test0['f1']:.4f} | "
    f"TTE={test0['tte_mae']:.4f}s | "
    f"speed={test0['speed_mae']:.4f}m/s | "
    f"heading={test0['heading_mae']:.3f}deg"
)


best_val = val0["loss"]
bad_epochs = 0


torch.save(
    {
        "epoch": 0,
        "model": model.state_dict(),
        "val_stats": val0,
    },
    BEST
)


for epoch in range(1, EPOCHS + 1):

    start = time.time()

    model.train()

    train_sum = 0.0
    seen = 0

    for batch in train_loader:

        batch = move(batch)

        optimizer.zero_grad(
            set_to_none=True
        )

        out = model(
            batch["agents"],
            batch["agent_mask"],
            batch["entry_line"],
            batch["interaction"],
        )

        loss = loss_fn(
            out,
            batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        optimizer.step()

        n = len(batch["decision"])

        train_sum += loss.item() * n
        seen += n


    val = evaluate(val_loader)

    sec = time.time() - start

    alpha = torch.tanh(
        model.interaction_alpha
    ).item()


    print(
        f"Epoch {epoch:02d} | "
        f"{sec:6.1f}s | "
        f"train={train_sum/seen:.4f} | "
        f"val={val['loss']:.4f} | "
        f"acc={val['accuracy']*100:6.2f}% | "
        f"F1={val['f1']:.4f} | "
        f"TTE={val['tte_mae']:.3f}s | "
        f"speed={val['speed_mae']:.3f}m/s | "
        f"heading={val['heading_mae']:.2f}deg | "
        f"alpha={alpha:+.4f}"
    )


    if val["loss"] < best_val:

        best_val = val["loss"]
        bad_epochs = 0

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "val_stats": val,
                "alpha": alpha,
            },
            BEST
        )

    else:

        bad_epochs += 1


    if bad_epochs >= PATIENCE:

        print()
        print("Early stopping.")
        break


best = torch.load(
    BEST,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    best["model"]
)

test = evaluate(test_loader)

alpha = torch.tanh(
    model.interaction_alpha
).item()


print()
print("=" * 110)
print("TEST BEST MAMBA RESIDUAL INTERACTION")
print("=" * 110)

print("Best epoch      :", best["epoch"])
print(f"Alpha           : {alpha:+.6f}")
print(f"Accuracy        : {test['accuracy']*100:.2f}%")
print(f"Precision       : {test['precision']:.4f}")
print(f"Recall          : {test['recall']:.4f}")
print(f"F1              : {test['f1']:.4f}")
print(f"TTE MAE         : {test['tte_mae']:.4f} s")
print(f"Entry speed MAE : {test['speed_mae']:.4f} m/s")
print(f"Heading MAE     : {test['heading_mae']:.3f} deg")

print()
print("Confusion matrix:")
print()
print("              Pred WAIT   Pred GO")
print(f"True WAIT     {test['tn']:9d} {test['fp']:9d}")
print(f"True GO       {test['fn']:9d} {test['tp']:9d}")

print()
print("Checkpoint:", BEST)
print("=" * 110)
