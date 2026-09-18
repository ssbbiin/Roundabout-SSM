from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "roundabout_ssm"

MATCH_FILE = (
    ROOT / "data/processed/sample_conflict_vehicles.csv"
)

DATA = (
    ROOT / "data/raw/data"
)

OUT_FILE = (
    ROOT / "data/processed/sample_interaction_features.csv"
)

# numerical safeguards / feature caps
MIN_CLOSING_SPEED = 0.3
MAX_EGO_TTC = 10.0

# simple current conflict-zone occupancy definition
CONFLICT_ZONE_RADIUS = 4.0


print("=" * 110)
print("COMPUTE INTERACTION FEATURES")
print("=" * 110)


matches = pd.read_csv(
    MATCH_FILE
)

results = []


for rec_id, rec_samples in matches.groupby(
    "recordingId"
):

    rec_id = int(rec_id)
    rid = f"{rec_id:02d}"

    print(
        f"Processing recording {rid} | "
        f"samples={len(rec_samples):,}"
    )

    # Current Ego state.
    # Rename immediately so there is no collision
    # with lead-vehicle columns in MATCH_FILE.
    tracks = pd.read_csv(
        DATA / f"{rid}_tracks.csv",
        usecols=[
            "trackId",
            "frame",
            "xCenter",
            "yCenter",
            "xVelocity",
            "yVelocity",
        ]
    ).rename(
        columns={
            "frame": "currentFrame",
            "xCenter": "egoX",
            "yCenter": "egoY",
            "xVelocity": "egoVX",
            "yVelocity": "egoVY",
        }
    )

    d = rec_samples.merge(
        tracks,
        on=[
            "trackId",
            "currentFrame"
        ],
        how="left",
        validate="many_to_one"
    )

    # ------------------------------------------------------
    # Ego -> conflict geometry
    # ------------------------------------------------------

    dx = (
        d["conflictX"]
        - d["egoX"]
    )

    dy = (
        d["conflictY"]
        - d["egoY"]
    )

    distance = np.sqrt(
        dx ** 2
        + dy ** 2
    )

    ego_speed = np.sqrt(
        d["egoVX"] ** 2
        + d["egoVY"] ** 2
    )

    # unit vector from Ego toward conflict point
    ux = dx / np.maximum(
        distance,
        1e-6
    )

    uy = dy / np.maximum(
        distance,
        1e-6
    )

    # Current velocity component actually moving
    # toward the conflict point.
    closing_speed = (
        d["egoVX"] * ux
        +
        d["egoVY"] * uy
    )

    approach_alignment = (
        closing_speed
        /
        np.maximum(
            ego_speed,
            1e-6
        )
    )

    # Current-velocity-only ego TTC.
    # This is NOT the ground-truth timeToEntry.
    safe_closing = np.maximum(
        closing_speed,
        MIN_CLOSING_SPEED
    )

    ego_cv_ttc = (
        distance
        / safe_closing
    )

    ego_cv_ttc = np.minimum(
        ego_cv_ttc,
        MAX_EGO_TTC
    )


    # ------------------------------------------------------
    # Circulating lead information
    # ------------------------------------------------------

    lead_present = (
        d["leadTrackId"] != -1
    )

    lead_ttc = (
        d["leadConflictTTC"]
    )

    signed_gap = (
        lead_ttc
        - ego_cv_ttc
    )

    abs_gap = np.abs(
        signed_gap
    )

    # signedGap:
    #
    # > 0 : Ego CV estimate reaches conflict BEFORE lead
    # < 0 : Lead reaches conflict BEFORE Ego
    #
    # Magnitude = temporal separation.

    lead_arrives_first = (
        signed_gap < 0
    )

    conflict_occupied = (
        lead_present
        &
        (
            d["leadEuclideanDistance"]
            <= CONFLICT_ZONE_RADIUS
        )
    )


    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    out = pd.DataFrame({
        "split":
            d["split"],

        "recordingId":
            d["recordingId"].astype(int),

        "trackId":
            d["trackId"].astype(int),

        "currentFrame":
            d["currentFrame"].astype(int),

        "entryId":
            d["entryId"].astype(int),

        # labels retained for analysis only
        "decision":
            d["decision"].astype(int),

        "timeToEntry":
            d["timeToEntry"],

        # Ego current-state interaction features
        "egoDistToConflict":
            distance,

        "egoSpeed":
            ego_speed,

        "egoClosingSpeed":
            closing_speed,

        "egoApproachAlignment":
            approach_alignment,

        "egoCvTTC":
            ego_cv_ttc,

        # Lead / conflict information
        "leadPresent":
            lead_present.astype(int),

        "leadTrackId":
            d["leadTrackId"].astype(int),

        "leadSpeed":
            d["leadSpeed"],

        "leadTangentSpeed":
            d["leadTangentSpeed"],

        "leadArcDistance":
            d["leadArcDistance"],

        "leadConflictTTC":
            lead_ttc,

        "leadAlignment":
            d["leadAlignment"],

        # Interaction
        "signedTTCGap":
            signed_gap,

        "absTTCGap":
            abs_gap,

        "leadArrivesFirst":
            lead_arrives_first
            .fillna(False)
            .astype(int),

        "conflictOccupied":
            conflict_occupied.astype(int),
    })

    results.append(out)


df = pd.concat(
    results,
    ignore_index=True
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUT_FILE,
    index=False
)


# ============================================================
# QA / STATISTICS
# ============================================================

print()
print("=" * 110)
print("RESULT")
print("=" * 110)

print(
    "Samples:",
    f"{len(df):,}"
)

print(
    "Missing Ego rows:",
    int(
        df[
            "egoDistToConflict"
        ].isna().sum()
    )
)

print(
    "Lead present:",
    f"{df['leadPresent'].mean()*100:.2f}%"
)


print()
print("=== EGO CURRENT-VELOCITY TTC ===")

print(
    df["egoCvTTC"]
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


matched = df[
    df["leadPresent"] == 1
].copy()


print()
print("=== MATCHED INTERACTION BY DECISION ===")

summary = (
    matched
    .groupby("decision")
    .agg(
        samples=(
            "trackId",
            "count"
        ),

        ego_ttc_mean=(
            "egoCvTTC",
            "mean"
        ),

        ego_ttc_median=(
            "egoCvTTC",
            "median"
        ),

        lead_ttc_mean=(
            "leadConflictTTC",
            "mean"
        ),

        lead_ttc_median=(
            "leadConflictTTC",
            "median"
        ),

        signed_gap_mean=(
            "signedTTCGap",
            "mean"
        ),

        signed_gap_median=(
            "signedTTCGap",
            "median"
        ),

        abs_gap_mean=(
            "absTTCGap",
            "mean"
        ),

        abs_gap_median=(
            "absTTCGap",
            "median"
        ),

        conflict_occupancy_rate=(
            "conflictOccupied",
            "mean"
        ),
    )
)

summary[
    "conflict_occupancy_rate"
] *= 100

summary = summary.rename(
    index={
        0: "WAIT",
        1: "GO"
    }
)

print(
    summary.to_string()
)


print()
print("=== SIGNED TTC GAP DISTRIBUTION ===")

for decision, name in [
    (0, "WAIT"),
    (1, "GO")
]:

    x = matched.loc[
        matched["decision"] == decision,
        "signedTTCGap"
    ]

    print()
    print(name)

    print(
        x.describe(
            percentiles=[
                .10,
                .25,
                .50,
                .75,
                .90
            ]
        )
    )


print()
print("=== LEAD-PRESENT RATE BY DECISION ===")

print(
    df.groupby(
        "decision"
    )["leadPresent"]
    .mean()
    .mul(100)
    .rename(
        index={
            0: "WAIT",
            1: "GO"
        }
    )
)


print()
print(
    "Saved:",
    OUT_FILE
)

print("=" * 110)
