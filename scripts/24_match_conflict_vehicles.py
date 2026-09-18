from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path.home() / "roundabout_ssm"

SAMPLES_FILE = (
    ROOT / "data/processed/all_decision_samples.csv"
)

CONFLICT_FILE = (
    ROOT / "configs/location0_conflict_geometry.json"
)

CACHE_DIR = (
    ROOT / "data/cache"
)

OUT_FILE = (
    ROOT / "data/processed/sample_conflict_vehicles.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

# Roundabout lane ring tolerance.
# Vehicles too far from the ring are not considered circulating.
RING_TOLERANCE_M = 4.0

# Minimum tangential speed to consider a moving circulating vehicle.
MIN_TANGENTIAL_SPEED = 0.5

# Velocity direction must agree with roundabout circulation.
MIN_TANGENT_ALIGNMENT = 0.70

# Only vehicles that can reach the conflict point soon are relevant.
MAX_CONFLICT_TTC = 8.0

# Vehicle classes from recording cache:
# car, van, truck, bus, trailer
MOTOR_CLASS_IDS = {0, 1, 2, 3, 4}


# ============================================================
# LOAD STATIC CONFLICT GEOMETRY
# ============================================================

with open(CONFLICT_FILE, "r") as f:
    geom = json.load(f)

entries = {
    int(e["entryId"]): e
    for e in geom["entries"]
}

conflict_points = np.array(
    [
        entries[i]["conflictPoint"]
        for i in range(4)
    ],
    dtype=np.float64
)

# Data-derived circle center
center = conflict_points.mean(axis=0)

cx = float(center[0])
cy = float(center[1])

radii = np.sqrt(
    (conflict_points[:, 0] - cx) ** 2
    +
    (conflict_points[:, 1] - cy) ** 2
)

roundabout_radius = float(
    radii.mean()
)


# ============================================================
# Determine circulation direction automatically.
#
# cross(radial, tangent)
# > 0 -> counter-clockwise
# < 0 -> clockwise
# ============================================================

orientation_votes = []

for entry_id in range(4):

    e = entries[entry_id]

    px, py = e["conflictPoint"]
    tx, ty = e["circulationTangent"]

    rx = px - cx
    ry = py - cy

    cross = (
        rx * ty
        - ry * tx
    )

    orientation_votes.append(
        np.sign(cross)
    )

circulation_sign = (
    1
    if np.mean(orientation_votes) > 0
    else -1
)

circulation_name = (
    "CCW"
    if circulation_sign > 0
    else "CW"
)


print("=" * 110)
print("CONFLICT VEHICLE MATCHING")
print("=" * 110)

print(
    f"Roundabout center : "
    f"({cx:.3f}, {cy:.3f})"
)

print(
    f"Mean ring radius  : "
    f"{roundabout_radius:.3f} m"
)

print(
    f"Circulation       : "
    f"{circulation_name}"
)

print(
    f"Ring tolerance    : "
    f"±{RING_TOLERANCE_M:.1f} m"
)

print(
    f"Max conflict TTC  : "
    f"{MAX_CONFLICT_TTC:.1f} s"
)

print()


# ============================================================
# HELPERS
# ============================================================

def wrap_2pi(angle):
    return angle % (2.0 * math.pi)


def angle_to_conflict(
    theta_vehicle,
    theta_conflict,
):
    """
    Angular distance from current vehicle position
    to conflict point following roundabout circulation.
    """

    if circulation_sign > 0:
        # counter-clockwise
        delta = (
            theta_conflict
            - theta_vehicle
        )
    else:
        # clockwise
        delta = (
            theta_vehicle
            - theta_conflict
        )

    return wrap_2pi(delta)


def load_recording(rec_id):

    path = (
        CACHE_DIR /
        f"{int(rec_id):02d}.npz"
    )

    z = np.load(
        path,
        allow_pickle=False
    )

    rec = {
        "track_ids":
            z["track_ids"],

        "frames":
            z["frames"],

        "states":
            z["states"],

        "unique_track_ids":
            z["unique_track_ids"],

        "track_class":
            z["track_class"],

        "unique_frames":
            z["unique_frames"],

        "frame_ptr":
            z["frame_ptr"],

        "frame_row_indices":
            z["frame_row_indices"],
    }

    z.close()

    return rec


def track_class_lookup(rec):

    return {
        int(track_id): int(cls)
        for track_id, cls
        in zip(
            rec["unique_track_ids"],
            rec["track_class"]
        )
    }


def rows_at_frame(
    rec,
    frame
):
    """
    Return row indices from track-sorted arrays
    for all agents existing at a given frame.
    """

    unique_frames = rec[
        "unique_frames"
    ]

    idx = np.searchsorted(
        unique_frames,
        frame
    )

    if (
        idx >= len(unique_frames)
        or unique_frames[idx] != frame
    ):
        return np.array(
            [],
            dtype=np.int64
        )

    start = rec["frame_ptr"][idx]
    end = rec["frame_ptr"][idx + 1]

    return (
        rec["frame_row_indices"][
            start:end
        ]
    )


# Cache state column indices
#
# states:
# 0 xCenter
# 1 yCenter
# 2 heading
# 3 xVelocity
# 4 yVelocity
# 5 xAcceleration
# 6 yAcceleration
X = 0
Y = 1
VX = 3
VY = 4


# ============================================================
# MATCH
# ============================================================

samples = pd.read_csv(
    SAMPLES_FILE
)

outputs = []


for rec_id, rec_samples in samples.groupby(
    "recordingId"
):

    rec_id = int(rec_id)

    print(
        f"Processing recording "
        f"{rec_id:02d} | "
        f"samples={len(rec_samples):,}"
    )

    rec = load_recording(
        rec_id
    )

    class_map = track_class_lookup(
        rec
    )

    # Avoid running identical frame searches repeatedly.
    frame_cache = {}


    for _, sample in rec_samples.iterrows():

        ego_track_id = int(
            sample["trackId"]
        )

        current_frame = int(
            sample["currentFrame"]
        )

        entry_id = int(
            sample["entryId"]
        )

        cp_x, cp_y = entries[
            entry_id
        ]["conflictPoint"]

        theta_conflict = math.atan2(
            cp_y - cy,
            cp_x - cx
        )


        if current_frame not in frame_cache:

            frame_cache[
                current_frame
            ] = rows_at_frame(
                rec,
                current_frame
            )

        row_indices = frame_cache[
            current_frame
        ]


        candidates = []


        for row_idx in row_indices:

            track_id = int(
                rec["track_ids"][
                    row_idx
                ]
            )

            if track_id == ego_track_id:
                continue


            # ---------------------------------------------
            # Motor vehicles only
            # ---------------------------------------------

            cls = class_map.get(
                track_id,
                -1
            )

            if cls not in MOTOR_CLASS_IDS:
                continue


            state = rec["states"][
                row_idx
            ]

            x = float(
                state[X]
            )

            y = float(
                state[Y]
            )

            vx = float(
                state[VX]
            )

            vy = float(
                state[VY]
            )


            # ---------------------------------------------
            # Ring membership
            # ---------------------------------------------

            rx = x - cx
            ry = y - cy

            radius = math.hypot(
                rx,
                ry
            )

            radial_error = abs(
                radius
                - roundabout_radius
            )

            if (
                radial_error
                > RING_TOLERANCE_M
            ):
                continue


            # ---------------------------------------------
            # Current circulation tangent
            # ---------------------------------------------

            theta_vehicle = math.atan2(
                ry,
                rx
            )

            if circulation_sign > 0:
                # CCW tangent
                tangent_x = -math.sin(
                    theta_vehicle
                )

                tangent_y = math.cos(
                    theta_vehicle
                )

            else:
                # CW tangent
                tangent_x = math.sin(
                    theta_vehicle
                )

                tangent_y = -math.cos(
                    theta_vehicle
                )


            speed = math.hypot(
                vx,
                vy
            )

            if speed < 1e-6:
                continue


            tangent_speed = (
                vx * tangent_x
                +
                vy * tangent_y
            )

            alignment = (
                tangent_speed
                / speed
            )


            # Vehicle must actually be moving in
            # the roundabout circulation direction.
            if (
                tangent_speed
                < MIN_TANGENTIAL_SPEED
            ):
                continue

            if (
                alignment
                < MIN_TANGENT_ALIGNMENT
            ):
                continue


            # ---------------------------------------------
            # Arc distance to target conflict point
            # ---------------------------------------------

            delta_theta = angle_to_conflict(
                theta_vehicle,
                theta_conflict
            )

            arc_distance = (
                roundabout_radius
                * delta_theta
            )

            conflict_ttc = (
                arc_distance
                / tangent_speed
            )


            if (
                conflict_ttc
                > MAX_CONFLICT_TTC
            ):
                continue


            euclidean_distance = math.hypot(
                x - cp_x,
                y - cp_y
            )


            candidates.append({
                "trackId":
                    track_id,

                "x":
                    x,

                "y":
                    y,

                "vx":
                    vx,

                "vy":
                    vy,

                "speed":
                    speed,

                "radius":
                    radius,

                "radialError":
                    radial_error,

                "tangentSpeed":
                    tangent_speed,

                "alignment":
                    alignment,

                "arcDistance":
                    arc_distance,

                "euclideanDistance":
                    euclidean_distance,

                "conflictTTC":
                    conflict_ttc,
            })


        # ---------------------------------------------
        # Next circulating vehicle to reach conflict
        # ---------------------------------------------

        candidates.sort(
            key=lambda x:
            x["conflictTTC"]
        )


        if len(candidates) > 0:

            lead = candidates[0]

            lead_track_id = (
                lead["trackId"]
            )

        else:

            lead = None
            lead_track_id = -1


        output = {
            "split":
                sample["split"],

            "recordingId":
                rec_id,

            "trackId":
                ego_track_id,

            "currentFrame":
                current_frame,

            "entryId":
                entry_id,

            "decision":
                int(sample["decision"]),

            "timeToEntry":
                float(sample["timeToEntry"]),

            "conflictX":
                float(cp_x),

            "conflictY":
                float(cp_y),

            "numConflictCandidates":
                len(candidates),

            "leadTrackId":
                lead_track_id,
        }


        if lead is not None:

            output.update({
                "leadX":
                    lead["x"],

                "leadY":
                    lead["y"],

                "leadVX":
                    lead["vx"],

                "leadVY":
                    lead["vy"],

                "leadSpeed":
                    lead["speed"],

                "leadRadius":
                    lead["radius"],

                "leadRadialError":
                    lead["radialError"],

                "leadTangentSpeed":
                    lead["tangentSpeed"],

                "leadAlignment":
                    lead["alignment"],

                "leadArcDistance":
                    lead["arcDistance"],

                "leadEuclideanDistance":
                    lead[
                        "euclideanDistance"
                    ],

                "leadConflictTTC":
                    lead["conflictTTC"],
            })

        else:

            for col in [
                "leadX",
                "leadY",
                "leadVX",
                "leadVY",
                "leadSpeed",
                "leadRadius",
                "leadRadialError",
                "leadTangentSpeed",
                "leadAlignment",
                "leadArcDistance",
                "leadEuclideanDistance",
                "leadConflictTTC",
            ]:
                output[col] = np.nan


        outputs.append(
            output
        )


# ============================================================
# SAVE + SUMMARY
# ============================================================

df = pd.DataFrame(
    outputs
)

df.to_csv(
    OUT_FILE,
    index=False
)


matched = (
    df["leadTrackId"]
    != -1
)

matched_count = int(
    matched.sum()
)

coverage = (
    100.0
    * matched_count
    / len(df)
)


print()
print("=" * 110)
print("RESULT")
print("=" * 110)

print(
    f"Samples            : "
    f"{len(df):,}"
)

print(
    f"Matched            : "
    f"{matched_count:,}"
)

print(
    f"Coverage           : "
    f"{coverage:.2f}%"
)


print()
print("=== COVERAGE BY DECISION ===")

coverage_decision = (
    df.assign(
        matched=matched
    )
    .groupby(
        "decision"
    )["matched"]
    .agg([
        "count",
        "sum",
        "mean"
    ])
)

coverage_decision[
    "mean"
] *= 100

print(
    coverage_decision
    .rename(
        index={
            0: "WAIT",
            1: "GO"
        }
    )
    .to_string()
)


print()
print("=== LEAD CONFLICT TTC ===")

print(
    df.loc[
        matched,
        "leadConflictTTC"
    ]
    .describe(
        percentiles=[
            .10,
            .25,
            .50,
            .75,
            .90,
            .95,
            .99
        ]
    )
)


print()
print(
    "=== LEAD TTC BY GO / WAIT ==="
)

print(
    df.loc[
        matched
    ]
    .groupby(
        "decision"
    )[
        "leadConflictTTC"
    ]
    .describe()
    .rename(
        index={
            0: "WAIT",
            1: "GO"
        }
    )
)


print()
print(
    "=== CANDIDATES PER SAMPLE ==="
)

print(
    df[
        "numConflictCandidates"
    ]
    .describe()
)


print()
print(
    "=== EXAMPLE MATCHES ==="
)

cols = [
    "recordingId",
    "trackId",
    "currentFrame",
    "entryId",
    "decision",
    "timeToEntry",
    "leadTrackId",
    "leadConflictTTC",
    "leadArcDistance",
    "leadSpeed",
    "leadAlignment",
]

print(
    df.loc[
        matched,
        cols
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print()
print(
    "Saved:",
    OUT_FILE
)

print("=" * 110)
