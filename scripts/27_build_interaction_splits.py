from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home() / "roundabout_ssm"

INTERACTION_FILE = (
    ROOT / "data/processed/sample_interaction_features.csv"
)

SPLIT_DIR = ROOT / "data/splits"

OUT_DIR = ROOT / "data/interaction_splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLS = [
    "egoDistToConflict",
    "egoClosingSpeed",
    "egoApproachAlignment",
    "egoCvTTC",

    "leadPresent",
    "leadSpeed",
    "leadArcDistance",
    "leadConflictTTC",

    "signedTTCGap",
    "conflictOccupied",
]


interaction = pd.read_csv(
    INTERACTION_FILE
)


# ============================================================
# Fill missing lead features.
#
# Important:
# leadPresent=0 tells the model that these zeros represent
# "no relevant circulating lead vehicle", not a real zero-TTC.
# ============================================================

LEAD_COLS = [
    "leadSpeed",
    "leadArcDistance",
    "leadConflictTTC",
    "signedTTCGap",
]

interaction[
    LEAD_COLS
] = interaction[
    LEAD_COLS
].fillna(0.0)


# ============================================================
# Create train/val/test files in EXACT original sample order.
# ============================================================

KEYS = [
    "recordingId",
    "trackId",
    "currentFrame",
    "entryId",
]


print("=" * 100)
print("BUILD INTERACTION SPLITS")
print("=" * 100)


for split in [
    "train",
    "val",
    "test",
]:

    base = pd.read_csv(
        SPLIT_DIR /
        f"{split}_samples.csv"
    )

    inter = interaction[
        interaction["split"] == split
    ].copy()

    # Only use keys + model features from interaction table.
    inter = inter[
        KEYS + FEATURE_COLS
    ]

    merged = base.merge(
        inter,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    # Should be exactly the original split size.
    assert len(merged) == len(base)

    missing = (
        merged[
            FEATURE_COLS
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    if missing != 0:
        raise RuntimeError(
            f"{split}: {missing} rows have missing interaction features"
        )

    out = (
        OUT_DIR /
        f"{split}_samples_interaction.csv"
    )

    merged.to_csv(
        out,
        index=False
    )

    print(
        f"{split:5s}: "
        f"{len(merged):6d} samples | "
        f"GO={merged['decision'].mean()*100:5.2f}% | "
        f"lead={merged['leadPresent'].mean()*100:5.2f}%"
    )

    print(
        f"       saved: {out}"
    )


# ============================================================
# Train-only normalization stats
# ============================================================

train = pd.read_csv(
    OUT_DIR /
    "train_samples_interaction.csv"
)

means = (
    train[FEATURE_COLS]
    .mean()
)

stds = (
    train[FEATURE_COLS]
    .std()
    .replace(0, 1)
)


stats = pd.DataFrame({
    "feature": FEATURE_COLS,
    "mean": [
        means[c]
        for c in FEATURE_COLS
    ],
    "std": [
        stds[c]
        for c in FEATURE_COLS
    ],
})


stats_file = (
    ROOT /
    "configs/interaction_normalization.csv"
)

stats.to_csv(
    stats_file,
    index=False
)


print()
print("=== TRAIN NORMALIZATION ===")
print(
    stats.to_string(
        index=False
    )
)

print()
print(
    "Saved normalization:",
    stats_file
)

print("=" * 100)
