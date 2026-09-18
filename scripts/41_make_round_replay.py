from pathlib import Path
import sys
import json
import math

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from matplotlib.animation import (
    FuncAnimation,
    PillowWriter
)

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.trajectory_dataset import (
    RoundaboutTrajectoryDataset
)

from models.mamba_trajectory import (
    MambaTrajectoryModel
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

RECORDING_ID = 5
TRACK_ID = 395

SCALE_DOWN_FACTOR = 10

CKPT = (
    ROOT /
    "outputs/mamba_trajectory/best.pt"
)

DATA_DIR = (
    ROOT /
    "data/raw/data"
)

CONFLICT_FILE = (
    ROOT /
    "configs/location0_conflict_geometry.json"
)

ENTRY_FILE = (
    ROOT /
    "configs/location0_geometry.json"
)

OUT = (
    ROOT /
    "outputs/round_replay_rec05_track395.gif"
)


# ============================================================
# DATASET
# ============================================================

ds = RoundaboutTrajectoryDataset(
    split="test",
    future_sec=4.0,
    future_step_sec=0.2,
)

table = ds.base.interaction_df

indices = np.where(
    (
        table["recordingId"].to_numpy()
        == RECORDING_ID
    )
    &
    (
        table["trackId"].to_numpy()
        == TRACK_ID
    )
)[0]

if len(indices) == 0:
    raise RuntimeError(
        f"No test samples for "
        f"recording={RECORDING_ID}, "
        f"track={TRACK_ID}"
    )

# chronological order
indices = sorted(
    indices,
    key=lambda i:
    int(
        table.iloc[i][
            "currentFrame"
        ]
    )
)

print("=" * 100)
print("ROUND OFFLINE REPLAY")
print("=" * 100)
print("Recording:", RECORDING_ID)
print("Track    :", TRACK_ID)
print("Frames   :", len(indices))
print()


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
# GEOMETRY
# ============================================================

with open(
    CONFLICT_FILE,
    "r"
) as f:
    conflict_geom = json.load(f)

with open(
    ENTRY_FILE,
    "r"
) as f:
    entry_geom = json.load(f)

conflict_map = {
    int(e["entryId"]): e
    for e in conflict_geom["entries"]
}

entry_map = {
    int(e["id"]): e
    for e in entry_geom["entries"]
}


# ============================================================
# BACKGROUND / RECORDING META
# ============================================================

rid = f"{RECORDING_ID:02d}"

background = plt.imread(
    DATA_DIR /
    f"{rid}_background.png"
)

recording_meta = pd.read_csv(
    DATA_DIR /
    f"{rid}_recordingMeta.csv"
).iloc[0]

ortho_scale = float(
    recording_meta[
        "orthoPxToMeter"
    ]
)


# ============================================================
# COORDINATE HELPERS
# ============================================================

def ego_to_world(
    points,
    ego_x,
    ego_y,
    heading_deg,
):
    points = np.asarray(
        points,
        dtype=np.float64
    )

    theta = np.deg2rad(
        heading_deg
    )

    c = np.cos(theta)
    s = np.sin(theta)

    x = points[:, 0]
    y = points[:, 1]

    world_x = (
        ego_x
        + c * x
        - s * y
    )

    world_y = (
        ego_y
        + s * x
        + c * y
    )

    return np.stack(
        [
            world_x,
            world_y
        ],
        axis=1
    )


def world_to_pixel(
    points
):
    points = np.asarray(
        points,
        dtype=np.float64
    )

    px = (
        points[:, 0]
        /
        (
            ortho_scale
            * SCALE_DOWN_FACTOR
        )
    )

    py = (
        -points[:, 1]
        /
        (
            ortho_scale
            * SCALE_DOWN_FACTOR
        )
    )

    return np.stack(
        [px, py],
        axis=1
    )


# ============================================================
# RECORDING CACHE
# ============================================================

core = ds.core

rec = core._load_recording(
    RECORDING_ID
)


def get_ego_pose(
    track_id,
    frame
):
    track_idx = (
        core._find_track_index(
            rec,
            track_id
        )
    )

    start = int(
        rec["track_ptr"][
            track_idx
        ]
    )

    end = int(
        rec["track_ptr"][
            track_idx + 1
        ]
    )

    frames = rec[
        "frames"
    ][start:end]

    states = rec[
        "states"
    ][start:end]

    j = np.searchsorted(
        frames,
        frame
    )

    if (
        j >= len(frames)
        or frames[j] != frame
    ):
        raise RuntimeError(
            "Ego frame missing."
        )

    state = states[j]

    return (
        float(
            state[core.X]
        ),
        float(
            state[core.Y]
        ),
        float(
            state[core.HEADING]
        ),
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

@torch.no_grad()
def predict(sample):

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

    prob_go = float(
        torch.softmax(
            out[
                "decision_logits"
            ],
            dim=1
        )[0, 1].item()
    )

    decision = int(
        out[
            "decision_logits"
        ]
        .argmax(dim=1)
        .item()
    )

    trajectory = (
        out[
            "future_trajectory"
        ][0]
        .cpu()
        .numpy()
    )

    return {
        "prob_go":
            prob_go,

        "decision":
            decision,

        "tte":
            float(
                out[
                    "time_to_entry"
                ][0].item()
            ),

        "trajectory":
            trajectory,
    }


# ============================================================
# PRECOMPUTE REPLAY FRAMES
# ============================================================

frames_data = []


for sample_idx in indices:

    sample = ds[
        sample_idx
    ]

    frame = int(
        sample[
            "current_frame"
        ]
    )

    entry_id = int(
        sample[
            "entry_id"
        ]
    )

    ego_x, ego_y, ego_heading = (
        get_ego_pose(
            TRACK_ID,
            frame
        )
    )

    pred = predict(
        sample
    )


    # --------------------------------------------------------
    # Past
    # --------------------------------------------------------

    ego_history = (
        sample[
            "agents"
        ][0]
        .numpy()
    )

    hist_mask = (
        ego_history[:, 8]
        > 0
    )

    past_relative = (
        ego_history[
            hist_mask,
            :2
        ]
    )

    past_world = ego_to_world(
        past_relative,
        ego_x,
        ego_y,
        ego_heading,
    )

    past_px = world_to_pixel(
        past_world
    )


    # --------------------------------------------------------
    # GT future
    # --------------------------------------------------------

    gt_relative = (
        sample[
            "future_trajectory"
        ]
        .numpy()
    )

    future_mask = (
        sample[
            "future_mask"
        ]
        .numpy()
        > 0
    )

    gt_world = ego_to_world(
        gt_relative[
            future_mask
        ],
        ego_x,
        ego_y,
        ego_heading,
    )

    gt_px = world_to_pixel(
        gt_world
    )


    # --------------------------------------------------------
    # Pred future
    # --------------------------------------------------------

    pred_world = ego_to_world(
        pred[
            "trajectory"
        ][future_mask],
        ego_x,
        ego_y,
        ego_heading,
    )

    pred_px = world_to_pixel(
        pred_world
    )


    # --------------------------------------------------------
    # Entry line
    # --------------------------------------------------------

    e = entry_map[
        entry_id
    ]

    entry_world = np.array(
        [
            e["p1_world"],
            e["p2_world"]
        ],
        dtype=np.float64
    )

    entry_px = world_to_pixel(
        entry_world
    )


    # --------------------------------------------------------
    # Conflict point
    # --------------------------------------------------------

    cp = np.array(
        [
            conflict_map[
                entry_id
            ]["conflictPoint"]
        ],
        dtype=np.float64
    )

    cp_px = world_to_pixel(
        cp
    )[0]


    # --------------------------------------------------------
    # All visible agents at current frame
    # --------------------------------------------------------

    ids, states = (
        core._agents_at_frame(
            rec,
            frame
        )
    )

    agent_world = states[
        :,
        [core.X, core.Y]
    ]

    agent_px = world_to_pixel(
        agent_world
    )


    # --------------------------------------------------------
    # Ego current pixel
    # --------------------------------------------------------

    ego_px = world_to_pixel(
        np.array(
            [[ego_x, ego_y]]
        )
    )[0]


    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    dist = np.sqrt(
        np.sum(
            (
                pred[
                    "trajectory"
                ][future_mask]
                -
                gt_relative[
                    future_mask
                ]
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


    raw_interaction = (
        sample[
            "interaction_raw"
        ]
        .numpy()
    )

    lead_present = bool(
        raw_interaction[4]
        > 0.5
    )

    if lead_present:
        lead_ttc = float(
            raw_interaction[7]
        )

        signed_gap = float(
            raw_interaction[8]
        )

    else:
        lead_ttc = None
        signed_gap = None


    frames_data.append({
        "frame":
            frame,

        "entry_id":
            entry_id,

        "agent_px":
            agent_px,

        "ego_px":
            ego_px,

        "past_px":
            past_px,

        "gt_px":
            gt_px,

        "pred_px":
            pred_px,

        "entry_px":
            entry_px,

        "conflict_px":
            cp_px,

        "gt_decision":
            int(
                sample[
                    "decision"
                ]
            ),

        "pred_decision":
            pred[
                "decision"
            ],

        "prob_go":
            pred[
                "prob_go"
            ],

        "gt_tte":
            float(
                sample[
                    "time_to_entry"
                ]
            ),

        "pred_tte":
            pred[
                "tte"
            ],

        "lead_present":
            lead_present,

        "lead_ttc":
            lead_ttc,

        "signed_gap":
            signed_gap,

        "ade":
            ade,

        "fde":
            fde,
    })


# ============================================================
# ANIMATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 9)
)


def update(i):

    d = frames_data[i]

    ax.clear()

    ax.imshow(
        background
    )


    # All agents
    ax.scatter(
        d["agent_px"][:, 0],
        d["agent_px"][:, 1],
        s=14,
        alpha=0.40,
        label="Current traffic"
    )


    # Entry
    ax.plot(
        d["entry_px"][:, 0],
        d["entry_px"][:, 1],
        linewidth=4,
        label="Entry line"
    )


    # Conflict
    ax.scatter(
        d["conflict_px"][0],
        d["conflict_px"][1],
        marker="X",
        s=150,
        label="Conflict point"
    )


    # Past
    ax.plot(
        d["past_px"][:, 0],
        d["past_px"][:, 1],
        linewidth=3,
        label="Ego past 2s"
    )


    # GT
    ax.plot(
        d["gt_px"][:, 0],
        d["gt_px"][:, 1],
        linewidth=2.5,
        label="GT future 4s"
    )


    # Prediction
    ax.plot(
        d["pred_px"][:, 0],
        d["pred_px"][:, 1],
        linestyle="--",
        linewidth=2.5,
        label="Mamba prediction"
    )


    # Ego
    ax.scatter(
        d["ego_px"][0],
        d["ego_px"][1],
        marker="o",
        s=160,
        label="Ego"
    )


    gt_dec = (
        "GO"
        if d[
            "gt_decision"
        ] == 1
        else "WAIT"
    )

    pred_dec = (
        "GO"
        if d[
            "pred_decision"
        ] == 1
        else "WAIT"
    )


    if d[
        "lead_present"
    ]:

        interaction_text = (
            f"Lead TTC: "
            f"{d['lead_ttc']:.2f} s\n"
            f"TTC gap: "
            f"{d['signed_gap']:+.2f} s"
        )

    else:

        interaction_text = (
            "Lead conflict vehicle: none"
        )


    text = (
        f"Recording {RECORDING_ID:02d} | "
        f"Track {TRACK_ID} | "
        f"Frame {d['frame']}\n\n"

        f"GT decision   : {gt_dec}\n"
        f"Pred decision : {pred_dec}\n"
        f"P(GO)         : {d['prob_go']:.3f}\n\n"

        f"GT TTE        : {d['gt_tte']:.2f} s\n"
        f"Pred TTE      : {d['pred_tte']:.2f} s\n\n"

        f"{interaction_text}\n\n"

        f"ADE           : {d['ade']:.2f} m\n"
        f"FDE           : {d['fde']:.2f} m"
    )


    ax.text(
        0.015,
        0.985,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(
            facecolor="white",
            alpha=0.88
        )
    )


    ax.set_title(
        "SSM-based Roundabout Entry Decision & Driving Prediction"
    )

    ax.axis(
        "off"
    )

    ax.legend(
        loc="lower center",
        ncol=4,
        fontsize=8
    )


ani = FuncAnimation(
    fig,
    update,
    frames=len(frames_data),
    interval=200,
    repeat=True
)


writer = PillowWriter(
    fps=5
)

ani.save(
    OUT,
    writer=writer,
    dpi=130
)

plt.close(fig)


print()
print("=" * 100)
print("REPLAY COMPLETE")
print("=" * 100)

print(
    "Frames :",
    len(frames_data)
)

print(
    "Output :",
    OUT
)

print("=" * 100)
