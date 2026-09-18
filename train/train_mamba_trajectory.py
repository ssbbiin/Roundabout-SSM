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

from dataset.trajectory_dataset import RoundaboutTrajectoryDataset
from models.mamba_trajectory import MambaTrajectoryModel


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 64
NUM_WORKERS = 2

EPOCHS = 15
PATIENCE = 4

LR = 1e-3
WEIGHT_DECAY = 1e-4

TRAJ_SCALE = 30.0  # meters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = ROOT / "outputs/mamba_trajectory"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST = OUT_DIR / "best.pt"

BASE_CKPT = (
    ROOT /
    "outputs/mamba_residual_interaction/best.pt"
)


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATA
# ============================================================

print("=" * 110)
print("MAMBA2 FUTURE TRAJECTORY HEAD TRAINING")
print("=" * 110)
print("Device:", DEVICE)


train_ds = RoundaboutTrajectoryDataset(
    split="train",
    future_sec=4.0,
    future_step_sec=0.2,
)

val_ds = RoundaboutTrajectoryDataset(
    split="val",
    future_sec=4.0,
    future_step_sec=0.2,
)

test_ds = RoundaboutTrajectoryDataset(
    split="test",
    future_sec=4.0,
    future_step_sec=0.2,
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

model = MambaTrajectoryModel(
    future_steps=20
).to(DEVICE)


base = torch.load(
    BASE_CKPT,
    map_location=DEVICE,
    weights_only=False
)

result = model.load_state_dict(
    base["model"],
    strict=False
)

print()
print("Loaded base checkpoint:", BASE_CKPT)
print("Missing keys:", len(result.missing_keys))
print("Unexpected keys:", len(result.unexpected_keys))

print(
    "Interaction alpha:",
    float(
        torch.tanh(
            model.interaction_alpha
        ).item()
    )
)

print(
    "Parameters:",
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# FREEZE EXISTING MODEL
#
# Only trajectory_head learns.
# Decision/TTE/speed/heading model remains exactly preserved.
# ============================================================

for p in model.parameters():
    p.requires_grad = False

for p in model.trajectory_head.parameters():
    p.requires_grad = True


trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Trainable trajectory params:",
    f"{trainable_params:,}"
)


optimizer = torch.optim.AdamW(
    model.trajectory_head.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# HELPERS
# ============================================================

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
        "future_trajectory",
        "future_mask",
    ]:
        batch[key] = batch[key].to(
            DEVICE,
            non_blocking=True
        )

    return batch


def trajectory_loss(
    pred,
    target,
    mask,
):
    """
    Smooth L1 in normalized coordinates.
    pred/target: [B,T,2]
    mask:        [B,T]
    """

    valid = mask.unsqueeze(-1)

    element_loss = F.smooth_l1_loss(
        pred / TRAJ_SCALE,
        target / TRAJ_SCALE,
        reduction="none"
    )

    element_loss = (
        element_loss
        * valid
    )

    denom = (
        valid.sum()
        * 2.0
    ).clamp_min(1.0)

    return (
        element_loss.sum()
        / denom
    )


def ade_fde(
    pred,
    target,
    mask,
):
    """
    Returns:
        total distance sum for ADE
        valid waypoint count
        FDE sum
        trajectories with >=1 valid waypoint
    """

    distance = torch.sqrt(
        (
            (pred - target) ** 2
        ).sum(dim=-1)
    )

    valid = mask > 0

    ade_sum = (
        distance[valid]
        .sum()
        .item()
    )

    ade_count = int(
        valid.sum().item()
    )


    # Last valid future waypoint for each sample
    fde_sum = 0.0
    fde_count = 0

    for i in range(
        len(pred)
    ):

        valid_idx = torch.where(
            valid[i]
        )[0]

        if len(valid_idx) == 0:
            continue

        last = int(
            valid_idx[-1]
        )

        fde_sum += float(
            distance[i, last].item()
        )

        fde_count += 1


    return (
        ade_sum,
        ade_count,
        fde_sum,
        fde_count
    )


@torch.no_grad()
def evaluate(dl):

    model.eval()

    loss_sum = 0.0
    samples = 0

    ade_sum = 0.0
    ade_count = 0

    fde_sum = 0.0
    fde_count = 0

    # Existing heads should stay unchanged.
    tp = tn = fp = fn = 0

    tte_abs = 0.0
    speed_abs = 0.0
    heading_abs = 0.0


    for batch in dl:

        batch = move(
            batch
        )

        out = model(
            batch["agents"],
            batch["agent_mask"],
            batch["entry_line"],
            batch["interaction"],
        )

        loss = trajectory_loss(
            out["future_trajectory"],
            batch["future_trajectory"],
            batch["future_mask"],
        )

        n = len(
            batch["decision"]
        )

        samples += n

        loss_sum += (
            loss.item()
            * n
        )


        # ------------------------------------------
        # Trajectory metrics
        # ------------------------------------------

        a_sum, a_count, f_sum, f_count = ade_fde(
            out["future_trajectory"],
            batch["future_trajectory"],
            batch["future_mask"],
        )

        ade_sum += a_sum
        ade_count += a_count

        fde_sum += f_sum
        fde_count += f_count


        # ------------------------------------------
        # Preserve existing prediction metrics
        # ------------------------------------------

        pred_decision = (
            out["decision_logits"]
            .argmax(dim=1)
        )

        gt = batch[
            "decision"
        ]

        tp += (
            (pred_decision == 1)
            & (gt == 1)
        ).sum().item()

        tn += (
            (pred_decision == 0)
            & (gt == 0)
        ).sum().item()

        fp += (
            (pred_decision == 1)
            & (gt == 0)
        ).sum().item()

        fn += (
            (pred_decision == 0)
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


        pred_h = F.normalize(
            out["entry_heading"],
            dim=1,
            eps=1e-8
        )

        true_h = F.normalize(
            batch["entry_heading"],
            dim=1,
            eps=1e-8
        )

        pred_angle = torch.atan2(
            pred_h[:, 0],
            pred_h[:, 1]
        )

        true_angle = torch.atan2(
            true_h[:, 0],
            true_h[:, 1]
        )

        diff = torch.atan2(
            torch.sin(
                pred_angle
                - true_angle
            ),
            torch.cos(
                pred_angle
                - true_angle
            )
        )

        heading_abs += (
            torch.abs(diff)
            * 180.0
            / math.pi
        ).sum().item()


    accuracy = (
        tp + tn
    ) / samples

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )


    return {
        "loss":
            loss_sum / samples,

        "ade":
            ade_sum / ade_count,

        "fde":
            fde_sum / fde_count,

        "accuracy":
            accuracy,

        "f1":
            f1,

        "tte_mae":
            tte_abs / samples,

        "speed_mae":
            speed_abs / samples,

        "heading_mae":
            heading_abs / samples,
    }


# ============================================================
# TRAIN
# ============================================================

best_val_ade = float(
    "inf"
)

bad_epochs = 0


for epoch in range(
    1,
    EPOCHS + 1
):

    start = time.time()

    model.train()

    # Frozen Mamba still contains dropout in scene encoder.
    # Disable stochastic layers in the frozen backbone,
    # while keeping trajectory head trainable.
    model.eval()
    model.trajectory_head.train()


    running = 0.0
    seen = 0


    for batch in train_loader:

        batch = move(
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

        loss = trajectory_loss(
            out["future_trajectory"],
            batch["future_trajectory"],
            batch["future_mask"],
        )

        loss.backward()

        optimizer.step()

        n = len(
            batch["decision"]
        )

        running += (
            loss.item()
            * n
        )

        seen += n


    val = evaluate(
        val_loader
    )

    sec = (
        time.time()
        - start
    )


    print(
        f"Epoch {epoch:02d} | "
        f"{sec:6.1f}s | "
        f"train={running/seen:.5f} | "
        f"val={val['loss']:.5f} | "
        f"ADE={val['ade']:.3f}m | "
        f"FDE={val['fde']:.3f}m | "
        f"F1={val['f1']:.4f} | "
        f"TTE={val['tte_mae']:.3f}s"
    )


    if (
        val["ade"]
        < best_val_ade
    ):

        best_val_ade = (
            val["ade"]
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
            },
            BEST
        )

    else:

        bad_epochs += 1


    if (
        bad_epochs
        >= PATIENCE
    ):

        print()
        print("Early stopping.")
        break


# ============================================================
# TEST BEST
# ============================================================

best = torch.load(
    BEST,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    best["model"]
)

test = evaluate(
    test_loader
)


print()
print("=" * 110)
print("TEST FINAL MAMBA TRAJECTORY MODEL")
print("=" * 110)

print(
    "Best epoch      :",
    best["epoch"]
)

print(
    f"Accuracy        : "
    f"{test['accuracy']*100:.2f}%"
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
    f"{test['heading_mae']:.3f} deg"
)

print()
print(
    f"Trajectory ADE  : "
    f"{test['ade']:.4f} m"
)

print(
    f"Trajectory FDE  : "
    f"{test['fde']:.4f} m"
)

print()
print(
    "Checkpoint:",
    BEST
)

print("=" * 110)
