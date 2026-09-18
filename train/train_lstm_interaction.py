from pathlib import Path
import sys
import time
import random
import csv
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.interaction_dataset import RoundaboutInteractionDataset
from models.lstm_interaction import LSTMInteraction


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 256
NUM_WORKERS = 2

EPOCHS = 25
PATIENCE = 6

LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

OUT_DIR = ROOT / "outputs/lstm_interaction_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT = OUT_DIR / "best.pt"
HISTORY_CSV = OUT_DIR / "history.csv"


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# DATASET
# ============================================================

print("=" * 100)
print("LSTM INTERACTION V2 TRAINING")
print("=" * 100)

print("Device:", DEVICE)

train_ds = RoundaboutInteractionDataset(
    split="train",
    history_frames=50,
    max_neighbors=8,
    neighbor_radius=30.0,
    max_cached_recordings=32,
)

val_ds = RoundaboutInteractionDataset(
    split="val",
    history_frames=50,
    max_neighbors=8,
    neighbor_radius=30.0,
    max_cached_recordings=32,
)

test_ds = RoundaboutInteractionDataset(
    split="test",
    history_frames=50,
    max_neighbors=8,
    neighbor_radius=30.0,
    max_cached_recordings=32,
)

print("Train:", len(train_ds))
print("Val  :", len(val_ds))
print("Test :", len(test_ds))


def make_loader(ds, shuffle):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


train_loader = make_loader(
    train_ds,
    True
)

val_loader = make_loader(
    val_ds,
    False
)

test_loader = make_loader(
    test_ds,
    False
)


# ============================================================
# MODEL
# ============================================================

model = LSTMInteraction(
    feature_dim=9,
    step_embed_dim=32,
    hidden_dim=64,
    lstm_layers=2,
    dropout=0.1,
).to(DEVICE)


num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: {num_params:,}"
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-5,
)


# ============================================================
# LOSS
# ============================================================

def compute_loss(
    output,
    batch,
):
    cls_loss = F.cross_entropy(
        output["decision_logits"],
        batch["decision"]
    )

    tte_loss = F.smooth_l1_loss(
        output["time_to_entry"] / 4.0,
        batch["time_to_entry"] / 4.0
    )

    speed_loss = F.smooth_l1_loss(
        output["entry_speed"] / 10.0,
        batch["entry_speed"] / 10.0
    )

    heading_loss = F.mse_loss(
        output["entry_heading"],
        batch["entry_heading"]
    )

    total = (
        cls_loss
        + 1.0 * tte_loss
        + 0.5 * speed_loss
        + 0.2 * heading_loss
    )

    return {
        "total": total,
        "cls": cls_loss,
        "tte": tte_loss,
        "speed": speed_loss,
        "heading": heading_loss,
    }


# ============================================================
# DEVICE TRANSFER
# ============================================================

def move_batch(batch):

    keys = [
        "agents",
        "agent_mask",
        "entry_line",
        "decision",
        "time_to_entry",
        "entry_speed",
        "entry_heading",
        "interaction",
    ]

    for key in keys:

        batch[key] = batch[key].to(
            DEVICE,
            non_blocking=True
        )

    return batch


# ============================================================
# METRICS
# ============================================================

def heading_error_deg(
    prediction,
    target
):
    # Normalize vectors first.
    prediction = F.normalize(
        prediction,
        dim=1,
        eps=1e-8
    )

    target = F.normalize(
        target,
        dim=1,
        eps=1e-8
    )

    pred_angle = torch.atan2(
        prediction[:, 0],
        prediction[:, 1]
    )

    true_angle = torch.atan2(
        target[:, 0],
        target[:, 1]
    )

    diff = torch.atan2(
        torch.sin(
            pred_angle - true_angle
        ),
        torch.cos(
            pred_angle - true_angle
        )
    )

    return (
        torch.abs(diff)
        * 180.0
        / math.pi
    )


@torch.no_grad()
def evaluate(loader):

    model.eval()

    total_samples = 0

    total_loss = 0.0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    tte_abs = 0.0
    speed_abs = 0.0
    heading_abs = 0.0

    for batch in loader:

        batch = move_batch(batch)

        output = model(
            batch["agents"],
            batch["agent_mask"],
            batch["entry_line"],
            batch["interaction"],
        )

        losses = compute_loss(
            output,
            batch
        )

        pred = (
            output["decision_logits"]
            .argmax(dim=1)
        )

        gt = batch["decision"]

        n = len(gt)

        total_samples += n

        total_loss += (
            losses["total"].item()
            * n
        )

        tp += (
            (pred == 1)
            & (gt == 1)
        ).sum().item()

        tn += (
            (pred == 0)
            & (gt == 0)
        ).sum().item()

        fp += (
            (pred == 1)
            & (gt == 0)
        ).sum().item()

        fn += (
            (pred == 0)
            & (gt == 1)
        ).sum().item()

        tte_abs += torch.abs(
            output["time_to_entry"]
            - batch["time_to_entry"]
        ).sum().item()

        speed_abs += torch.abs(
            output["entry_speed"]
            - batch["entry_speed"]
        ).sum().item()

        heading_abs += (
            heading_error_deg(
                output["entry_heading"],
                batch["entry_heading"]
            )
            .sum()
            .item()
        )

    accuracy = (
        tp + tn
    ) / total_samples

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
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "loss":
            total_loss / total_samples,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "tte_mae":
            tte_abs / total_samples,

        "speed_mae":
            speed_abs / total_samples,

        "heading_mae_deg":
            heading_abs / total_samples,

        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# HISTORY FILE
# ============================================================

with open(
    HISTORY_CSV,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_tte_mae",
        "val_speed_mae",
        "val_heading_mae_deg",
        "lr",
        "epoch_sec"
    ])


# ============================================================
# TRAIN
# ============================================================

best_val_loss = float("inf")
bad_epochs = 0


for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start = time.time()

    model.train()

    running_loss = 0.0
    seen = 0

    for batch in train_loader:

        batch = move_batch(batch)

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            batch["agents"],
            batch["agent_mask"],
            batch["entry_line"],
            batch["interaction"],
        )

        losses = compute_loss(
            output,
            batch
        )

        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        optimizer.step()

        n = batch[
            "agents"
        ].shape[0]

        running_loss += (
            losses["total"].item()
            * n
        )

        seen += n


    train_loss = (
        running_loss / seen
    )


    val_stats = evaluate(
        val_loader
    )


    scheduler.step(
        val_stats["loss"]
    )


    epoch_sec = (
        time.time()
        - epoch_start
    )

    lr = optimizer.param_groups[0]["lr"]


    print(
        f"Epoch {epoch:02d} | "
        f"{epoch_sec:6.1f}s | "
        f"train={train_loss:.4f} | "
        f"val={val_stats['loss']:.4f} | "
        f"acc={val_stats['accuracy']*100:6.2f}% | "
        f"F1={val_stats['f1']:.4f} | "
        f"TTE={val_stats['tte_mae']:.3f}s | "
        f"speed={val_stats['speed_mae']:.3f}m/s | "
        f"heading={val_stats['heading_mae_deg']:.2f}deg | "
        f"lr={lr:.2e}"
    )


    with open(
        HISTORY_CSV,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            epoch,
            train_loss,
            val_stats["loss"],
            val_stats["accuracy"],
            val_stats["precision"],
            val_stats["recall"],
            val_stats["f1"],
            val_stats["tte_mae"],
            val_stats["speed_mae"],
            val_stats["heading_mae_deg"],
            lr,
            epoch_sec
        ])


    # ------------------------------------------
    # checkpoint
    # ------------------------------------------

    if (
        val_stats["loss"]
        < best_val_loss
    ):

        best_val_loss = (
            val_stats["loss"]
        )

        bad_epochs = 0

        torch.save(
            {
                "epoch":
                    epoch,

                "model":
                    model.state_dict(),

                "optimizer":
                    optimizer.state_dict(),

                "val_stats":
                    val_stats,

                "config": {
                    "feature_dim": 9,
                    "step_embed_dim": 32,
                    "hidden_dim": 64,
                    "lstm_layers": 2,
                    "batch_size": BATCH_SIZE,
                    "history_frames": 50,
                    "max_neighbors": 8,
                    "neighbor_radius": 30.0,
                }
            },
            BEST_CKPT
        )

    else:

        bad_epochs += 1


    if bad_epochs >= PATIENCE:

        print()
        print(
            f"Early stopping after "
            f"{PATIENCE} epochs "
            f"without improvement."
        )

        break


# ============================================================
# TEST BEST CHECKPOINT
# ============================================================

print()
print("=" * 100)
print("TEST BEST CHECKPOINT")
print("=" * 100)

checkpoint = torch.load(
    BEST_CKPT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model"]
)

test_stats = evaluate(
    test_loader
)


print(
    f"Best epoch      : "
    f"{checkpoint['epoch']}"
)

print(
    f"Accuracy        : "
    f"{test_stats['accuracy']*100:.2f}%"
)

print(
    f"Precision       : "
    f"{test_stats['precision']:.4f}"
)

print(
    f"Recall          : "
    f"{test_stats['recall']:.4f}"
)

print(
    f"F1              : "
    f"{test_stats['f1']:.4f}"
)

print(
    f"TTE MAE         : "
    f"{test_stats['tte_mae']:.4f} s"
)

print(
    f"Entry speed MAE : "
    f"{test_stats['speed_mae']:.4f} m/s"
)

print(
    f"Heading MAE     : "
    f"{test_stats['heading_mae_deg']:.3f} deg"
)

print()
print("Confusion matrix:")
print()
print("              Pred WAIT   Pred GO")
print(
    f"True WAIT     "
    f"{test_stats['tn']:9d} "
    f"{test_stats['fp']:9d}"
)

print(
    f"True GO       "
    f"{test_stats['fn']:9d} "
    f"{test_stats['tp']:9d}"
)

print()
print("Checkpoint:", BEST_CKPT)
print("History   :", HISTORY_CSV)
print("=" * 100)
