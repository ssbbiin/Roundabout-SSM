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

from dataset.interaction_dataset import (
    RoundaboutInteractionDataset
)

from models.lstm_residual_interaction import (
    LSTMResidualInteraction
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 256
NUM_WORKERS = 2

EPOCHS = 15
PATIENCE = 5

BASE_LR = 5e-5
INTERACTION_LR = 5e-4

WEIGHT_DECAY = 1e-4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

OUT_DIR = (
    ROOT /
    "outputs/lstm_residual_interaction"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_CKPT = (
    OUT_DIR /
    "best.pt"
)

HISTORY_CSV = (
    OUT_DIR /
    "history.csv"
)

V1_CKPT = (
    ROOT /
    "outputs/lstm_baseline/best.pt"
)


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
# DATA
# ============================================================

print("=" * 110)
print("LSTM RESIDUAL INTERACTION TRAINING")
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
# MODEL + LOAD V1
# ============================================================

model = LSTMResidualInteraction(
    feature_dim=9,
    step_embed_dim=32,
    hidden_dim=64,
    lstm_layers=2,
    dropout=0.1,
    interaction_dim=10,
).to(DEVICE)


v1 = torch.load(
    V1_CKPT,
    map_location=DEVICE,
    weights_only=False
)

load_result = model.load_state_dict(
    v1["model"],
    strict=False
)


print()
print(
    "Loaded v1 checkpoint:",
    V1_CKPT
)

print(
    "Missing new keys:",
    len(load_result.missing_keys)
)

print(
    "Unexpected keys:",
    len(load_result.unexpected_keys)
)


num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Parameters:",
    f"{num_params:,}"
)


# ============================================================
# PARAMETER GROUPS
# ============================================================

interaction_params = []
base_params = []

for name, p in model.named_parameters():

    if (
        name.startswith(
            "interaction_encoder"
        )
        or name
        == "interaction_alpha"
    ):
        interaction_params.append(p)

    else:
        base_params.append(p)


optimizer = torch.optim.AdamW(
    [
        {
            "params":
                base_params,

            "lr":
                BASE_LR,
        },

        {
            "params":
                interaction_params,

            "lr":
                INTERACTION_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
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
        + tte_loss
        + 0.5 * speed_loss
        + 0.2 * heading_loss
    )

    return total


def move_batch(batch):

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

        batch[key] = (
            batch[key]
            .to(
                DEVICE,
                non_blocking=True
            )
        )

    return batch


# ============================================================
# METRICS
# ============================================================

def heading_error_deg(
    prediction,
    target,
):

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

    pa = torch.atan2(
        prediction[:, 0],
        prediction[:, 1]
    )

    ta = torch.atan2(
        target[:, 0],
        target[:, 1]
    )

    diff = torch.atan2(
        torch.sin(pa - ta),
        torch.cos(pa - ta)
    )

    return (
        torch.abs(diff)
        * 180.0
        / math.pi
    )


@torch.no_grad()
def evaluate(loader):

    model.eval()

    total = 0
    loss_sum = 0.0

    tp = tn = fp = fn = 0

    tte_abs = 0.0
    speed_abs = 0.0
    heading_abs = 0.0


    for batch in loader:

        batch = move_batch(
            batch
        )

        out = model(
            batch["agents"],
            batch["agent_mask"],
            batch["entry_line"],
            batch["interaction"],
        )

        loss = compute_loss(
            out,
            batch
        )

        pred = (
            out["decision_logits"]
            .argmax(dim=1)
        )

        gt = batch[
            "decision"
        ]

        n = len(gt)

        total += n

        loss_sum += (
            loss.item()
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
            out["time_to_entry"]
            - batch["time_to_entry"]
        ).sum().item()

        speed_abs += torch.abs(
            out["entry_speed"]
            - batch["entry_speed"]
        ).sum().item()

        heading_abs += (
            heading_error_deg(
                out["entry_heading"],
                batch["entry_heading"]
            )
            .sum()
            .item()
        )


    accuracy = (
        tp + tn
    ) / total

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )


    return {
        "loss":
            loss_sum / total,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "tte_mae":
            tte_abs / total,

        "speed_mae":
            speed_abs / total,

        "heading_mae_deg":
            heading_abs / total,

        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# EPOCH 0 = EXACT V1 BEHAVIOR
# ============================================================

initial_val = evaluate(
    val_loader
)

initial_test = evaluate(
    test_loader
)


print()
print("=" * 110)
print("EPOCH 0 — V1 INITIALIZATION")
print("=" * 110)

print(
    f"alpha="
    f"{torch.tanh(model.interaction_alpha).item():+.6f}"
)

print(
    f"VAL  | "
    f"loss={initial_val['loss']:.4f} | "
    f"F1={initial_val['f1']:.4f} | "
    f"TTE={initial_val['tte_mae']:.4f}s"
)

print(
    f"TEST | "
    f"acc={initial_test['accuracy']*100:.2f}% | "
    f"F1={initial_test['f1']:.4f} | "
    f"TTE={initial_test['tte_mae']:.4f}s | "
    f"speed={initial_test['speed_mae']:.4f}m/s | "
    f"heading={initial_test['heading_mae_deg']:.3f}deg"
)


# Initial best IS v1.
best_val_loss = (
    initial_val[
        "loss"
    ]
)

bad_epochs = 0


torch.save(
    {
        "epoch": 0,
        "model":
            model.state_dict(),

        "val_stats":
            initial_val,

        "source":
            "v1 initialization",
    },
    BEST_CKPT
)


# ============================================================
# HISTORY
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
        "val_f1",
        "val_tte_mae",
        "val_speed_mae",
        "val_heading_mae_deg",
        "alpha",
        "epoch_sec",
    ])


# ============================================================
# TRAIN
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    start = time.time()

    model.train()

    running = 0.0
    seen = 0


    for batch in train_loader:

        batch = move_batch(
            batch
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        out = model(
            batch["agents"],
            batch["agent_mask"],
            batch["entry_line"],
            batch["interaction"],
        )

        loss = compute_loss(
            out,
            batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        optimizer.step()

        n = len(
            batch["decision"]
        )

        running += (
            loss.item() * n
        )

        seen += n


    train_loss = (
        running / seen
    )

    val = evaluate(
        val_loader
    )

    epoch_sec = (
        time.time()
        - start
    )

    alpha = torch.tanh(
        model.interaction_alpha
    ).item()


    print(
        f"Epoch {epoch:02d} | "
        f"{epoch_sec:5.1f}s | "
        f"train={train_loss:.4f} | "
        f"val={val['loss']:.4f} | "
        f"acc={val['accuracy']*100:6.2f}% | "
        f"F1={val['f1']:.4f} | "
        f"TTE={val['tte_mae']:.3f}s | "
        f"speed={val['speed_mae']:.3f}m/s | "
        f"heading={val['heading_mae_deg']:.2f}deg | "
        f"alpha={alpha:+.4f}"
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
            val["loss"],
            val["accuracy"],
            val["f1"],
            val["tte_mae"],
            val["speed_mae"],
            val["heading_mae_deg"],
            alpha,
            epoch_sec,
        ])


    if (
        val["loss"]
        < best_val_loss
    ):

        best_val_loss = (
            val["loss"]
        )

        bad_epochs = 0

        torch.save(
            {
                "epoch":
                    epoch,

                "model":
                    model.state_dict(),

                "val_stats":
                    val,

                "alpha":
                    alpha,
            },
            BEST_CKPT
        )

    else:

        bad_epochs += 1


    if bad_epochs >= PATIENCE:

        print()
        print(
            "Early stopping."
        )

        break


# ============================================================
# BEST TEST
# ============================================================

best = torch.load(
    BEST_CKPT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    best["model"]
)

test = evaluate(
    test_loader
)

alpha = torch.tanh(
    model.interaction_alpha
).item()


print()
print("=" * 110)
print("TEST BEST RESIDUAL INTERACTION")
print("=" * 110)

print(
    "Best epoch      :",
    best["epoch"]
)

print(
    f"Alpha           : "
    f"{alpha:+.6f}"
)

print(
    f"Accuracy        : "
    f"{test['accuracy']*100:.2f}%"
)

print(
    f"Precision       : "
    f"{test['precision']:.4f}"
)

print(
    f"Recall          : "
    f"{test['recall']:.4f}"
)

print(
    f"F1              : "
    f"{test['f1']:.4f}"
)

print(
    f"TTE MAE         : "
    f"{test['tte_mae']:.4f} s"
)

print(
    f"Entry speed MAE : "
    f"{test['speed_mae']:.4f} m/s"
)

print(
    f"Heading MAE     : "
    f"{test['heading_mae_deg']:.3f} deg"
)

print()
print("Confusion matrix:")
print()
print(
    "              "
    "Pred WAIT   Pred GO"
)

print(
    f"True WAIT     "
    f"{test['tn']:9d} "
    f"{test['fp']:9d}"
)

print(
    f"True GO       "
    f"{test['fn']:9d} "
    f"{test['tp']:9d}"
)

print()
print(
    "Checkpoint:",
    BEST_CKPT
)

print("=" * 110)
