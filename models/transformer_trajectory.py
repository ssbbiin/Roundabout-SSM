import torch
import torch.nn as nn

from models.transformer_baseline import TransformerBaseline


class TransformerTrajectoryModel(TransformerBaseline):
    def __init__(self, future_steps=20):
        super().__init__()

        self.future_steps = future_steps
        self._scene_feature = None

        # Capture the existing 64-dimensional scene representation.
        # Original Transformer predictions are unchanged.
        self.scene_encoder.register_forward_hook(
            self._capture_scene
        )

        # Same trajectory-head architecture as LSTM and Mamba2.
        self.trajectory_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, future_steps * 2),
        )

    def _capture_scene(self, module, inputs, output):
        self._scene_feature = output

    def forward(self, agents, agent_mask, entry_line):
        self._scene_feature = None

        result = super().forward(
            agents,
            agent_mask,
            entry_line
        )

        scene = self._scene_feature
        self._scene_feature = None

        if scene is None:
            raise RuntimeError("Scene feature was not captured.")

        result["future_trajectory"] = (
            self.trajectory_head(scene)
            .reshape(-1, self.future_steps, 2)
        )

        return result
