from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.roundabout_dataset import RoundaboutDatasetCore


class RoundaboutTorchDataset(Dataset):

    def __init__(
        self,
        split="train",
        history_frames=50,
        max_neighbors=8,
        neighbor_radius=30.0,
        max_cached_recordings=32,
    ):
        self.core = RoundaboutDatasetCore(
            split=split,
            history_frames=history_frames,
            max_neighbors=max_neighbors,
            neighbor_radius=neighbor_radius,
            max_cached_recordings=max_cached_recordings,
        )

        self.max_neighbors = max_neighbors

    def __len__(self):
        return len(self.core)

    def __getitem__(self, index):

        s = self.core[index]

        target = s["target"]
        meta = s["metadata"]

        # neighbor ID도 고정 길이로 만들어둠
        neighbor_ids = np.full(
            self.max_neighbors,
            -1,
            dtype=np.int64
        )

        ids = meta["neighbor_ids"]

        if len(ids) > 0:
            neighbor_ids[:len(ids)] = np.asarray(
                ids,
                dtype=np.int64
            )

        return {
            "agents": torch.from_numpy(
                s["agents"]
            ).float(),

            "agent_mask": torch.from_numpy(
                s["agent_mask"]
            ).float(),

            "entry_line": torch.from_numpy(
                s["entry_line"]
            ).float(),

            "decision": torch.tensor(
                target["decision"],
                dtype=torch.long
            ),

            "time_to_entry": torch.tensor(
                target["time_to_entry"],
                dtype=torch.float32
            ),

            "entry_speed": torch.tensor(
                target["entry_speed"],
                dtype=torch.float32
            ),

            "entry_heading": torch.tensor(
                [
                    target["entry_heading_sin"],
                    target["entry_heading_cos"]
                ],
                dtype=torch.float32
            ),

            # fixed-size metadata
            "recording_id": torch.tensor(
                meta["recording_id"],
                dtype=torch.long
            ),

            "track_id": torch.tensor(
                meta["track_id"],
                dtype=torch.long
            ),

            "entry_id": torch.tensor(
                meta["entry_id"],
                dtype=torch.long
            ),

            "current_frame": torch.tensor(
                meta["current_frame"],
                dtype=torch.long
            ),

            "neighbor_ids": torch.from_numpy(
                neighbor_ids
            )
        }
