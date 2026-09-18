from pathlib import Path
from collections import OrderedDict
import json
import numpy as np
import pandas as pd


class RoundaboutDatasetCore:
    """
    rounD roundabout sample extractor.

    Output:
        agents      : [9, 50, 9]
                      agent 0 = Ego
                      agent 1~8 = nearest neighbors

        agent_mask  : [9]
        entry_line  : [2, 2] in current Ego coordinate
        target      : decision / TTE / entry speed /
                      relative entry heading sin, cos
    """

    MOTOR_CLASS_IDS = {0, 1, 2, 3, 4}

    # states[] column index from recording cache
    X = 0
    Y = 1
    HEADING = 2
    VX = 3
    VY = 4
    AX = 5
    AY = 6

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
            root = Path.home() / "roundabout_ssm"

        self.root = Path(root)

        self.cache_dir = self.root / "data/cache"
        self.sample_file = (
            self.root /
            f"data/splits/{split}_samples.csv"
        )

        self.geometry_file = (
            self.root /
            "configs/location0_geometry.json"
        )

        self.history_frames = history_frames
        self.max_neighbors = max_neighbors
        self.neighbor_radius = neighbor_radius

        self.max_cached_recordings = max_cached_recordings
        self.cache = OrderedDict()

        self.samples = pd.read_csv(self.sample_file)

        with open(self.geometry_file, "r") as f:
            self.geometry = json.load(f)

        self.entries = {
            int(e["id"]): e
            for e in self.geometry["entries"]
        }

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------
    # recording cache
    # ------------------------------------------------------

    def _load_recording(self, rec_id):
        rec_id = int(rec_id)

        if rec_id in self.cache:
            item = self.cache.pop(rec_id)
            self.cache[rec_id] = item
            return item

        path = self.cache_dir / f"{rec_id:02d}.npz"

        z = np.load(path, allow_pickle=False)

        item = {
            "track_ids": z["track_ids"],
            "frames": z["frames"],
            "states": z["states"],

            "unique_track_ids": z["unique_track_ids"],
            "track_ptr": z["track_ptr"],
            "track_class": z["track_class"],

            "unique_frames": z["unique_frames"],
            "frame_ptr": z["frame_ptr"],
            "frame_row_indices": z["frame_row_indices"],
        }

        z.close()

        self.cache[rec_id] = item

        while len(self.cache) > self.max_cached_recordings:
            self.cache.popitem(last=False)

        return item

    # ------------------------------------------------------
    # lookup helpers
    # ------------------------------------------------------

    @staticmethod
    def _find_track_index(rec, track_id):
        ids = rec["unique_track_ids"]

        idx = np.searchsorted(ids, track_id)

        if idx >= len(ids) or ids[idx] != track_id:
            return None

        return int(idx)

    @staticmethod
    def _find_frame_index(rec, frame):
        frames = rec["unique_frames"]

        idx = np.searchsorted(frames, frame)

        if idx >= len(frames) or frames[idx] != frame:
            return None

        return int(idx)

    def _get_track_history(
        self,
        rec,
        track_id,
        current_frame,
    ):
        """
        Return:
            states   [T, state_dim]
            presence [T]
        """

        track_idx = self._find_track_index(
            rec,
            track_id
        )

        T = self.history_frames

        output = np.zeros(
            (T, rec["states"].shape[1]),
            dtype=np.float32
        )

        presence = np.zeros(
            T,
            dtype=np.float32
        )

        if track_idx is None:
            return output, presence

        start = rec["track_ptr"][track_idx]
        end = rec["track_ptr"][track_idx + 1]

        frames = rec["frames"][start:end]
        states = rec["states"][start:end]

        history_start = (
            current_frame - T + 1
        )

        valid = (
            (frames >= history_start) &
            (frames <= current_frame)
        )

        if not np.any(valid):
            return output, presence

        f = frames[valid]
        s = states[valid]

        positions = (
            f - history_start
        ).astype(np.int64)

        valid_pos = (
            (positions >= 0) &
            (positions < T)
        )

        positions = positions[valid_pos]
        s = s[valid_pos]

        output[positions] = s
        presence[positions] = 1.0

        return output, presence

    # ------------------------------------------------------
    # agents present at current frame
    # ------------------------------------------------------

    def _agents_at_frame(
        self,
        rec,
        frame,
    ):
        frame_idx = self._find_frame_index(
            rec,
            frame
        )

        if frame_idx is None:
            return (
                np.empty(0, dtype=np.int32),
                np.empty((0, rec["states"].shape[1]),
                         dtype=np.float32)
            )

        start = rec["frame_ptr"][frame_idx]
        end = rec["frame_ptr"][frame_idx + 1]

        row_indices = (
            rec["frame_row_indices"][start:end]
        )

        return (
            rec["track_ids"][row_indices],
            rec["states"][row_indices]
        )

    def _track_class(self, rec, track_id):
        idx = self._find_track_index(
            rec,
            track_id
        )

        if idx is None:
            return -1

        return int(
            rec["track_class"][idx]
        )

    # ------------------------------------------------------
    # ego coordinate transform
    # ------------------------------------------------------

    @staticmethod
    def _world_to_ego_xy(
        x,
        y,
        ego_x,
        ego_y,
        ego_heading_deg,
    ):
        theta = np.deg2rad(
            ego_heading_deg
        )

        c = np.cos(theta)
        s = np.sin(theta)

        dx = x - ego_x
        dy = y - ego_y

        rel_x = c * dx + s * dy
        rel_y = -s * dx + c * dy

        return rel_x, rel_y

    @staticmethod
    def _rotate_world_vector(
        x,
        y,
        ego_heading_deg,
    ):
        theta = np.deg2rad(
            ego_heading_deg
        )

        c = np.cos(theta)
        s = np.sin(theta)

        rel_x = c * x + s * y
        rel_y = -s * x + c * y

        return rel_x, rel_y

    def _convert_history_to_features(
        self,
        states,
        presence,
        ego_x,
        ego_y,
        ego_heading,
    ):
        """
        Output feature:
        0 rel_x
        1 rel_y
        2 rel_vx
        3 rel_vy
        4 rel_ax
        5 rel_ay
        6 sin(relative_heading)
        7 cos(relative_heading)
        8 presence
        """

        T = len(states)

        feat = np.zeros(
            (T, 9),
            dtype=np.float32
        )

        valid = presence > 0

        if not np.any(valid):
            return feat

        st = states[valid]

        rel_x, rel_y = self._world_to_ego_xy(
            st[:, self.X],
            st[:, self.Y],
            ego_x,
            ego_y,
            ego_heading
        )

        rel_vx, rel_vy = self._rotate_world_vector(
            st[:, self.VX],
            st[:, self.VY],
            ego_heading
        )

        rel_ax, rel_ay = self._rotate_world_vector(
            st[:, self.AX],
            st[:, self.AY],
            ego_heading
        )

        heading_delta = np.deg2rad(
            st[:, self.HEADING] -
            ego_heading
        )

        feat[valid, 0] = rel_x
        feat[valid, 1] = rel_y

        feat[valid, 2] = rel_vx
        feat[valid, 3] = rel_vy

        feat[valid, 4] = rel_ax
        feat[valid, 5] = rel_ay

        feat[valid, 6] = np.sin(
            heading_delta
        )

        feat[valid, 7] = np.cos(
            heading_delta
        )

        feat[valid, 8] = 1.0

        return feat

    # ------------------------------------------------------
    # neighbor selection
    # ------------------------------------------------------

    def _select_neighbors(
        self,
        rec,
        ego_track_id,
        current_frame,
        ego_x,
        ego_y,
    ):
        track_ids, states = (
            self._agents_at_frame(
                rec,
                current_frame
            )
        )

        candidates = []

        for track_id, state in zip(
            track_ids,
            states
        ):
            track_id = int(track_id)

            if track_id == ego_track_id:
                continue

            cls = self._track_class(
                rec,
                track_id
            )

            if cls not in self.MOTOR_CLASS_IDS:
                continue

            dx = float(
                state[self.X] - ego_x
            )

            dy = float(
                state[self.Y] - ego_y
            )

            distance = np.hypot(
                dx,
                dy
            )

            if distance > self.neighbor_radius:
                continue

            candidates.append(
                (
                    float(distance),
                    track_id
                )
            )

        candidates.sort(
            key=lambda x: x[0]
        )

        return [
            track_id
            for _, track_id
            in candidates[:self.max_neighbors]
        ]

    # ------------------------------------------------------
    # entry geometry
    # ------------------------------------------------------

    def _entry_line_ego(
        self,
        entry_id,
        ego_x,
        ego_y,
        ego_heading,
    ):
        entry = self.entries[
            int(entry_id)
        ]

        p1 = entry["p1_world"]
        p2 = entry["p2_world"]

        p1x, p1y = self._world_to_ego_xy(
            p1[0],
            p1[1],
            ego_x,
            ego_y,
            ego_heading
        )

        p2x, p2y = self._world_to_ego_xy(
            p2[0],
            p2[1],
            ego_x,
            ego_y,
            ego_heading
        )

        return np.array(
            [
                [p1x, p1y],
                [p2x, p2y]
            ],
            dtype=np.float32
        )

    # ------------------------------------------------------
    # sample
    # ------------------------------------------------------

    def __getitem__(self, index):
        row = self.samples.iloc[index]

        rec_id = int(
            row["recordingId"]
        )

        ego_track_id = int(
            row["trackId"]
        )

        current_frame = int(
            row["currentFrame"]
        )

        entry_id = int(
            row["entryId"]
        )

        rec = self._load_recording(
            rec_id
        )

        # --------------------------------------------------
        # Ego history
        # --------------------------------------------------
        ego_states, ego_presence = (
            self._get_track_history(
                rec,
                ego_track_id,
                current_frame
            )
        )

        if ego_presence[-1] == 0:
            raise RuntimeError(
                f"Ego missing at current frame: "
                f"rec={rec_id}, "
                f"track={ego_track_id}, "
                f"frame={current_frame}"
            )

        ego_current = ego_states[-1]

        ego_x = float(
            ego_current[self.X]
        )

        ego_y = float(
            ego_current[self.Y]
        )

        ego_heading = float(
            ego_current[self.HEADING]
        )

        ego_feat = (
            self._convert_history_to_features(
                ego_states,
                ego_presence,
                ego_x,
                ego_y,
                ego_heading
            )
        )

        # --------------------------------------------------
        # Neighbors
        # --------------------------------------------------
        neighbor_ids = (
            self._select_neighbors(
                rec,
                ego_track_id,
                current_frame,
                ego_x,
                ego_y
            )
        )

        A = 1 + self.max_neighbors

        agents = np.zeros(
            (
                A,
                self.history_frames,
                9
            ),
            dtype=np.float32
        )

        agents[0] = ego_feat

        agent_mask = np.zeros(
            A,
            dtype=np.float32
        )

        agent_mask[0] = 1.0

        for i, neighbor_id in enumerate(
            neighbor_ids
        ):
            states, presence = (
                self._get_track_history(
                    rec,
                    neighbor_id,
                    current_frame
                )
            )

            feat = (
                self._convert_history_to_features(
                    states,
                    presence,
                    ego_x,
                    ego_y,
                    ego_heading
                )
            )

            agents[i + 1] = feat
            agent_mask[i + 1] = 1.0

        # --------------------------------------------------
        # Map geometry
        # --------------------------------------------------
        entry_line = (
            self._entry_line_ego(
                entry_id,
                ego_x,
                ego_y,
                ego_heading
            )
        )

        # --------------------------------------------------
        # Targets
        # --------------------------------------------------
        entry_heading = float(
            row["entryHeading"]
        )

        heading_delta = np.deg2rad(
            entry_heading - ego_heading
        )

        target = {
            "decision":
                np.int64(row["decision"]),

            "time_to_entry":
                np.float32(row["timeToEntry"]),

            "entry_speed":
                np.float32(row["entrySpeed"]),

            "entry_heading_sin":
                np.float32(
                    np.sin(heading_delta)
                ),

            "entry_heading_cos":
                np.float32(
                    np.cos(heading_delta)
                ),
        }

        metadata = {
            "recording_id": rec_id,
            "track_id": ego_track_id,
            "entry_id": entry_id,
            "current_frame": current_frame,
            "neighbor_ids": neighbor_ids,
        }

        return {
            "agents": agents,
            "agent_mask": agent_mask,
            "entry_line": entry_line,
            "target": target,
            "metadata": metadata,
        }
