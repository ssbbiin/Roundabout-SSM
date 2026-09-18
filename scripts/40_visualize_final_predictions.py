from pathlib import Path
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.trajectory_dataset import RoundaboutTrajectoryDataset
from models.mamba_trajectory import MambaTrajectoryModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = ROOT / "outputs/mamba_trajectory/best.pt"
CONFLICT_FILE = ROOT / "configs/location0_conflict_geometry.json"
OUT = ROOT / "outputs/final_mamba_test_predictions.png"


# ============================================================
# DATASET
# ============================================================

ds = RoundaboutTrajectoryDataset(
    split="test",
    future_sec=4.0,
    future_step_sec=0.2,
)

df = ds.base.interaction_df


# ============================================================
# MODEL
# ============================================================

model = MambaTrajectoryModel(
    future_steps=20
).to(DEVICE)

ckpt = torch.load(
    CKPT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    ckpt["model"]
)

model.eval()


# ============================================================
# CONFLICT GEOMETRY
# ============================================================

with open(CONFLICT_FILE, "r") as f:
    conflict_geom = json.load(f)

conflict_entries = {
    int(e["entryId"]): e
    for e in conflict_geom["entries"]
}


# ============================================================
# SELECT DIVERSE TEST EXAMPLES
# ============================================================

targets = [
    0.4,
    1.0,
    1.6,
    2.2,
    3.0,
    4.0,
]

indices = []

for target_tte in targets:

    candidates = np.where(
        np.isclose(
            df["timeToEntry"].to_numpy(),
            target_tte
        )
    )[0]

    if len(candidates) == 0:
        continue

    # deterministic
    idx = int(
        candidates[
            len(candidates) // 2
        ]
    )

    indices.append(idx)


# ============================================================
# HELPERS
# ============================================================

def predict(sample):

    with torch.no_grad():

        out = model(
            sample["agents"]
            .unsqueeze(0)
            .to(DEVICE),

            sample["agent_mask"]
            .unsqueeze(0)
            .to(DEVICE),

            sample["entry_line"]
            .unsqueeze(0)
            .to(DEVICE),

            sample["interaction"]
            .unsqueeze(0)
            .to(DEVICE),
        )

    result = {}

    result["decision_prob"] = (
        torch.softmax(
            out["decision_logits"],
            dim=1
        )[0, 1]
        .item()
    )

    result["decision"] = int(
        out["decision_logits"]
        .argmax(dim=1)
        .item()
    )

    result["tte"] = float(
        out["time_to_entry"][0]
        .item()
    )

    result["speed"] = float(
        out["entry_speed"][0]
        .item()
    )

    heading = out[
        "entry_heading"
    ][0]

    result["heading_delta_deg"] = float(
        torch.atan2(
            heading[0],
            heading[1]
        ).item()
        * 180.0
        / np.pi
    )

    result["trajectory"] = (
        out["future_trajectory"][0]
        .cpu()
        .numpy()
    )

    return result


def conflict_point_ego(sample):

    rec_id = int(
        sample["recording_id"]
    )

    track_id = int(
        sample["track_id"]
    )

    current_frame = int(
        sample["current_frame"]
    )

    entry_id = int(
        sample["entry_id"]
    )

    core = ds.core

    rec = core._load_recording(
        rec_id
    )

    track_idx = core._find_track_index(
        rec,
        track_id
    )

    start = int(
        rec["track_ptr"][track_idx]
    )

    end = int(
        rec["track_ptr"][track_idx + 1]
    )

    frames = rec["frames"][start:end]
    states = rec["states"][start:end]

    j = np.searchsorted(
        frames,
        current_frame
    )

    state = states[j]

    ego_x = float(
        state[core.X]
    )

    ego_y = float(
        state[core.Y]
    )

    ego_heading = float(
        state[core.HEADING]
    )

    cp = conflict_entries[
        entry_id
    ]["conflictPoint"]

    x, y = core._world_to_ego_xy(
        cp[0],
        cp[1],
        ego_x,
        ego_y,
        ego_heading,
    )

    return np.array(
        [x, y],
        dtype=np.float32
    )


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 11)
)

axes = axes.flatten()


for ax, idx in zip(
    axes,
    indices
):

    sample = ds[idx]

    pred = predict(
        sample
    )


    # --------------------------------------------------------
    # Ego history
    # --------------------------------------------------------

    ego = sample[
        "agents"
    ][0].numpy()

    valid_hist = (
        ego[:, 8] > 0
    )

    ax.plot(
        ego[valid_hist, 0],
        ego[valid_hist, 1],
        linewidth=3,
        label="Past 2s"
    )


    # --------------------------------------------------------
    # Ground truth future
    # --------------------------------------------------------

    gt = sample[
        "future_trajectory"
    ].numpy()

    future_mask = (
        sample[
            "future_mask"
        ].numpy()
        > 0
    )

    ax.plot(
        gt[future_mask, 0],
        gt[future_mask, 1],
        marker="o",
        markersize=3,
        linewidth=2,
        label="GT future"
    )


    # --------------------------------------------------------
    # Predicted future
    # --------------------------------------------------------

    pred_traj = pred[
        "trajectory"
    ]

    ax.plot(
        pred_traj[
            future_mask, 0
        ],
        pred_traj[
            future_mask, 1
        ],
        linestyle="--",
        marker="x",
        markersize=3,
        linewidth=2,
        label="Mamba prediction"
    )


    # --------------------------------------------------------
    # Entry line
    # --------------------------------------------------------

    entry = sample[
        "entry_line"
    ].numpy()

    ax.plot(
        entry[:, 0],
        entry[:, 1],
        linewidth=4,
        label="Entry line"
    )


    # --------------------------------------------------------
    # Conflict point
    # --------------------------------------------------------

    cp = conflict_point_ego(
        sample
    )

    ax.scatter(
        cp[0],
        cp[1],
        marker="X",
        s=110,
        label="Conflict point"
    )


    # --------------------------------------------------------
    # Current Ego
    # --------------------------------------------------------

    ax.scatter(
        0,
        0,
        marker="o",
        s=100,
        label="Current Ego"
    )

    ax.arrow(
        0,
        0,
        3,
        0,
        width=0.04,
        head_width=0.45,
        length_includes_head=True
    )


    # --------------------------------------------------------
    # Per-sample trajectory errors
    # --------------------------------------------------------

    dist = np.sqrt(
        np.sum(
            (
                pred_traj[future_mask]
                - gt[future_mask]
            ) ** 2,
            axis=1
        )
    )

    ade = float(
        dist.mean()
    )

    fde = float(
        dist[-1]
    )


    # --------------------------------------------------------
    # Labels / predictions
    # --------------------------------------------------------

    gt_decision = (
        "GO"
        if int(
            sample["decision"]
        ) == 1
        else "WAIT"
    )

    pred_decision = (
        "GO"
        if pred["decision"] == 1
        else "WAIT"
    )

    gt_tte = float(
        sample[
            "time_to_entry"
        ]
    )


    ax.set_title(
        f"Rec {int(sample['recording_id']):02d} | "
        f"Track {int(sample['track_id'])} | "
        f"Entry {int(sample['entry_id'])}\n"
        f"GT={gt_decision}, Pred={pred_decision} "
        f"(P(GO)={pred['decision_prob']:.3f}) | "
        f"TTE {gt_tte:.1f}s → {pred['tte']:.2f}s\n"
        f"ADE={ade:.2f}m | FDE={fde:.2f}m"
    )

    ax.set_xlabel(
        "Forward [m]"
    )

    ax.set_ylabel(
        "Left [m]"
    )

    ax.axhline(
        0,
        linewidth=0.5
    )

    ax.axvline(
        0,
        linewidth=0.5
    )

    ax.set_aspect(
        "equal"
    )

    ax.grid(True)


# shared legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=6
)

plt.tight_layout(
    rect=[0, 0.06, 1, 1]
)

plt.savefig(
    OUT,
    dpi=200,
    bbox_inches="tight"
)

print("=" * 100)
print("FINAL TEST PREDICTION VISUALIZATION")
print("=" * 100)
print("Checkpoint:", CKPT)
print("Output    :", OUT)
print("Samples   :", indices)
print("=" * 100)

plt.show()
