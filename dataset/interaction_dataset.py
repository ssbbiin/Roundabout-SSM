from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.torch_dataset import RoundaboutTorchDataset


class RoundaboutInteractionDataset(Dataset):

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

    KEY_COLS = [
        "recordingId",
        "trackId",
        "currentFrame",
        "entryId",
    ]

    def __init__(
        self,
        split="train",
        root=None,
        history_frames=50,
        max_neighbors=8,
        neighbor_radius=30.0,
        max_cached_recordings=32,
    ):

        if root is None:
            root = ROOT

        self.root = Path(root)
        self.split = split

        # Existing v1 trajectory dataset.
        self.base = RoundaboutTorchDataset(
            split=split,
            history_frames=history_frames,
            max_neighbors=max_neighbors,
            neighbor_radius=neighbor_radius,
            max_cached_recordings=max_cached_recordings,
        )

        interaction_file = (
            self.root
            / "data/interaction_splits"
            / f"{split}_samples_interaction.csv"
        )

        self.interaction_df = pd.read_csv(
            interaction_file
        )

        if len(self.interaction_df) != len(self.base):
            raise RuntimeError(
                f"Length mismatch: "
                f"base={len(self.base)}, "
                f"interaction={len(self.interaction_df)}"
            )

        # --------------------------------------------------
        # Verify that row order is exactly identical
        # to the original split.
        # --------------------------------------------------

        base_df = self.base.core.samples

        for key in self.KEY_COLS:

            a = base_df[key].to_numpy()
            b = self.interaction_df[key].to_numpy()

            if not np.array_equal(a, b):
                raise RuntimeError(
                    f"Row alignment mismatch in column: {key}"
                )

        # --------------------------------------------------
        # Train-only normalization parameters
        # --------------------------------------------------

        stats_file = (
            self.root
            / "configs/interaction_normalization.csv"
        )

        stats = pd.read_csv(
            stats_file
        ).set_index("feature")

        self.mean = np.array(
            [
                stats.loc[col, "mean"]
                for col in self.FEATURE_COLS
            ],
            dtype=np.float32
        )

        self.std = np.array(
            [
                stats.loc[col, "std"]
                for col in self.FEATURE_COLS
            ],
            dtype=np.float32
        )

        self.std = np.maximum(
            self.std,
            1e-6
        )

        # Raw interaction matrix
        self.raw_features = (
            self.interaction_df[
                self.FEATURE_COLS
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        if not np.isfinite(
            self.raw_features
        ).all():

            raise RuntimeError(
                "Non-finite interaction feature detected."
            )


    def __len__(self):
        return len(self.base)


    def __getitem__(self, index):

        sample = self.base[index]

        raw = self.raw_features[
            index
        ]

        normalized = (
            raw - self.mean
        ) / self.std

        # If no relevant circulating lead vehicle exists,
        # neutralize lead-derived features.
        #
        # indices:
        # 4 leadPresent
        # 5 leadSpeed
        # 6 leadArcDistance
        # 7 leadConflictTTC
        # 8 signedTTCGap
        # 9 conflictOccupied
        if raw[4] < 0.5:
            normalized[5:10] = 0.0

        sample[
            "interaction"
        ] = torch.from_numpy(
            normalized.copy()
        ).float()

        # QA/debug purposes only.
        sample[
            "interaction_raw"
        ] = torch.from_numpy(
            raw.copy()
        ).float()

        return sample
