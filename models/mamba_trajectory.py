import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba_residual_interaction import (
    MambaResidualInteraction
)


class MambaTrajectoryModel(MambaResidualInteraction):

    def __init__(
        self,
        feature_dim=9,
        d_model=64,
        d_state=64,
        num_layers=2,
        dropout=0.1,
        interaction_dim=10,
        future_steps=20,
    ):
        super().__init__(
            feature_dim=feature_dim,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            dropout=dropout,
            interaction_dim=interaction_dim,
        )

        self.future_steps = future_steps

        self.trajectory_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),

            nn.Linear(
                128,
                future_steps * 2
            )
        )


    def forward(
        self,
        agents,
        agent_mask,
        entry_line,
        interaction,
    ):

        B, A, T, Fdim = agents.shape

        # --------------------------------------------------
        # Mamba temporal branch
        # --------------------------------------------------

        x = (
            agents /
            self.feature_scale
        )

        x = x.reshape(
            B * A,
            T,
            Fdim
        )

        x = self.input_encoder(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        agent_embedding = (
            x[:, -1]
            .reshape(B, A, -1)
        )

        mask = (
            agent_mask.unsqueeze(-1)
        )

        mean_embedding = (
            agent_embedding * mask
        ).sum(dim=1)

        mean_embedding = (
            mean_embedding
            /
            mask.sum(dim=1)
            .clamp_min(1.0)
        )

        max_embedding = (
            agent_embedding
            .masked_fill(
                mask == 0,
                -1e9
            )
            .max(dim=1)
            .values
        )

        ego_embedding = (
            agent_embedding[:, 0]
        )

        # --------------------------------------------------
        # Entry geometry
        # --------------------------------------------------

        entry_embedding = (
            self.entry_encoder(
                entry_line.reshape(B, 4)
                / 30.0
            )
        )

        scene_input = torch.cat(
            [
                ego_embedding,
                mean_embedding,
                max_embedding,
                entry_embedding,
            ],
            dim=-1
        )

        base_scene = (
            self.scene_encoder(
                scene_input
            )
        )

        # --------------------------------------------------
        # Residual TTC / Gap interaction
        # --------------------------------------------------

        correction = (
            self.interaction_encoder(
                interaction
            )
        )

        alpha = torch.tanh(
            self.interaction_alpha
        )

        scene = (
            base_scene
            +
            alpha * correction
        )

        # --------------------------------------------------
        # Existing heads
        # --------------------------------------------------

        decision_logits = (
            self.decision_head(
                scene
            )
        )

        time_to_entry = F.softplus(
            self.tte_head(scene)
        ).squeeze(-1)

        entry_speed = F.softplus(
            self.speed_head(scene)
        ).squeeze(-1)

        entry_heading = (
            self.heading_head(
                scene
            )
        )

        # --------------------------------------------------
        # Future trajectory
        # [B, 20, 2]
        # --------------------------------------------------

        future_trajectory = (
            self.trajectory_head(
                scene
            )
            .reshape(
                B,
                self.future_steps,
                2
            )
        )

        return {
            "decision_logits":
                decision_logits,

            "time_to_entry":
                time_to_entry,

            "entry_speed":
                entry_speed,

            "entry_heading":
                entry_heading,

            "future_trajectory":
                future_trajectory,

            "interaction_alpha":
                alpha,
        }
