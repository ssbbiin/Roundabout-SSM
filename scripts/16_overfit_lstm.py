from pathlib import Path
import sys
import random
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    Subset
)

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.torch_dataset import (
    RoundaboutTorchDataset
)

from models.lstm_baseline import (
    LSTMBaseline
)


# ============================================================
# Config
# ============================================================

SEED = 42

NUM_SAMPLES = 1024
BATCH_SIZE = 128
EPOCHS = 50

LR = 3e-3
NUM_WORKERS = 2

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Seed
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Dataset
# ============================================================

print("=" * 90)
print("LSTM SANITY OVERFIT TEST")
print("=" * 90)
print("Device:", DEVICE)


full_ds = RoundaboutTorchDataset(
    split="train",
    history_frames=50,
    max_neighbors=8,
    neighbor_radius=30.0,
    max_cached_recordings=32,
)


rng = np.random.default_rng(
    SEED
)

indices = rng.choice(
    len(full_ds),
    size=NUM_SAMPLES,
    replace=False
)

subset = Subset(
    full_ds,
    indices.tolist()
)


loader = DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)


# ============================================================
# Model
# ============================================================

model = LSTMBaseline(
    feature_dim=9,
    step_embed_dim=32,
    hidden_dim=64,
    lstm_layers=2,
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
    weight_decay=1e-4
)


# ============================================================
# Loss
# ============================================================

def compute_loss(
    output,
    batch,
):
    # Classification
    cls_loss = F.cross_entropy(
        output["decision_logits"],
        batch["decision"]
    )

    # Normalize regression scale only inside loss
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

    return (
        total,
        cls_loss,
        tte_loss,
        speed_loss,
        heading_loss
    )


# ============================================================
# Evaluation on same 1024 samples
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    total = 0
    correct = 0

    tte_error = 0.0
    speed_error = 0.0

    total_loss = 0.0

    for batch in loader:

        for key in [
            "agents",
            "agent_mask",
            "entry_line",
            "decision",
            "time_to_entry",
            "entry_speed",
            "entry_heading",
        ]:
            batch[key] = batch[key].to(
                DEVICE,
                non_blocking=True
            )

        output = model(
            batch["agents"],
            batch["agent_mask"],
            batch["entry_line"],
        )

        loss, *_ = compute_loss(
            output,
            batch
        )

        pred = output[
            "decision_logits"
        ].argmax(dim=1)

        n = len(pred)

        total += n

        correct += (
            pred == batch["decision"]
        ).sum().item()

        tte_error += torch.abs(
            output["time_to_entry"]
            - batch["time_to_entry"]
        ).sum().item()

        speed_error += torch.abs(
            output["entry_speed"]
            - batch["entry_speed"]
        ).sum().item()

        total_loss += (
            loss.item() * n
        )

    return {
        "loss":
            total_loss / total,

        "accuracy":
            correct / total,

        "tte_mae":
            tte_error / total,

        "speed_mae":
            speed_error / total,
    }


# ============================================================
# Train
# ============================================================

best_loss = float("inf")

OUT = (
    ROOT /
    "outputs/lstm_sanity_best.pt"
)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running = 0.0
    seen = 0

    for batch in loader:

        for key in [
            "agents",
            "agent_mask",
            "entry_line",
            "decision",
            "time_to_entry",
            "entry_speed",
            "entry_heading",
        ]:
            batch[key] = batch[key].to(
                DEVICE,
                non_blocking=True
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            batch["agents"],
            batch["agent_mask"],
            batch["entry_line"],
        )

        loss, *_ = compute_loss(
            output,
            batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        n = batch[
            "agents"
        ].shape[0]

        running += (
            loss.item() * n
        )

        seen += n


    stats = evaluate()

    if stats["loss"] < best_loss:

        best_loss = stats["loss"]

        torch.save(
            {
                "model":
                    model.state_dict(),

                "epoch":
                    epoch,

                "stats":
                    stats,
            },
            OUT
        )


    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={running/seen:.4f} | "
        f"eval_loss={stats['loss']:.4f} | "
        f"acc={stats['accuracy']*100:6.2f}% | "
        f"TTE_MAE={stats['tte_mae']:.3f}s | "
        f"speed_MAE={stats['speed_mae']:.3f}m/s"
    )


    # Enough for sanity verification
    if (
        stats["accuracy"] >= 0.98
        and stats["tte_mae"] <= 0.15
        and stats["speed_mae"] <= 0.50
    ):

        print()
        print("SANITY TEST PASSED.")
        print(
            "Small subset can be overfit."
        )

        break


print()
print("=" * 90)
print("DONE")
print("=" * 90)
print(
    "Best checkpoint:",
    OUT
)
