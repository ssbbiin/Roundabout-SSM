from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.interaction_dataset import (
    RoundaboutInteractionDataset
)


class RoundaboutTrajectoryDataset(Dataset):

    def __init__(
        self,
        split="train",
        root=None,
        history_frames=50,
        max_neighbors=8,
        neighbor_radius=30.0,
        max_cached_recordings=32,
        future_sec=4.0,
        future_step_sec=0.2,
    ):

        if root is None:
            root = ROOT

        self.root = Path(root)

        self.future_sec = float(
            future_sec
        )

        self.future_step_sec = float(
            future_step_sec
        )

        self.num_future_steps = int(
            round(
                self.future_sec
                /
                self.future_step_sec
            )
        )

        self.base = RoundaboutInteractionDataset(
            split=split,
            root=root,
            history_frames=history_frames,
            max_neighbors=max_neighbors,
            neighbor_radius=neighbor_radius,
            max_cached_recordings=max_cached_recordings,
        )

        # shortcut to original rounD core
        self.core = (
            self.base
            .base
            .core
        )


    def __len__(self):
        return len(self.base)


    def _future_trajectory(
        self,
        index,
        rec_id,
        track_id,
        current_frame,
    ):

        rec = self.core._load_recording(
            rec_id
        )

        track_idx = (
            self.core._find_track_index(
                rec,
                track_id
            )
        )

        K = self.num_future_steps

        trajectory = np.zeros(
            (K, 2),
            dtype=np.float32
        )

        mask = np.zeros(
            K,
            dtype=np.float32
        )

        if track_idx is None:
            return trajectory, mask


        # --------------------------------------------------
        # Track arrays
        # --------------------------------------------------

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


        # --------------------------------------------------
        # Current Ego world pose
        # --------------------------------------------------

        pos = np.searchsorted(
            frames,
            current_frame
        )

        if (
            pos >= len(frames)
            or frames[pos]
            != current_frame
        ):
            raise RuntimeError(
                f"Current ego frame missing: "
                f"rec={rec_id}, "
                f"track={track_id}, "
                f"frame={current_frame}"
            )

        current_state = states[pos]

        ego_x = float(
            current_state[
                self.core.X
            ]
        )

        ego_y = float(
            current_state[
                self.core.Y
            ]
        )

        ego_heading = float(
            current_state[
                self.core.HEADING
            ]
        )


        # All recordings currently use 25 Hz,
        # but use the sample metadata rather than hardcoding.
        frame_rate = float(
            self.base
            .interaction_df
            .iloc[index][
                "frameRate"
            ]
        )


        # --------------------------------------------------
        # Future waypoints:
        # 0.2, 0.4, ..., 4.0 sec
        # --------------------------------------------------

        for k in range(
            1,
            K + 1
        ):

            future_time = (
                k
                * self.future_step_sec
            )

            offset = int(
                round(
                    future_time
                    * frame_rate
                )
            )

            target_frame = (
                current_frame
                + offset
            )

            j = np.searchsorted(
                frames,
                target_frame
            )

            if (
                j >= len(frames)
                or frames[j]
                != target_frame
            ):
                continue

            state = states[j]

            x = float(
                state[
                    self.core.X
                ]
            )

            y = float(
                state[
                    self.core.Y
                ]
            )

            rel_x, rel_y = (
                self.core
                ._world_to_ego_xy(
                    x,
                    y,
                    ego_x,
                    ego_y,
                    ego_heading,
                )
            )

            trajectory[
                k - 1,
                0
            ] = rel_x

            trajectory[
                k - 1,
                1
            ] = rel_y

            mask[
                k - 1
            ] = 1.0


        return (
            trajectory,
            mask
        )


    def __getitem__(self, index):

        sample = self.base[
            index
        ]

        rec_id = int(
            sample[
                "recording_id"
            ]
        )

        track_id = int(
            sample[
                "track_id"
            ]
        )

        current_frame = int(
            sample[
                "current_frame"
            ]
        )

        trajectory, mask = (
            self._future_trajectory(
                index,
                rec_id,
                track_id,
                current_frame,
            )
        )

        sample[
            "future_trajectory"
        ] = torch.from_numpy(
            trajectory
        ).float()

        sample[
            "future_mask"
        ] = torch.from_numpy(
            mask
        ).float()

        return sample
