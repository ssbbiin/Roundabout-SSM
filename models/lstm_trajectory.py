import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lstm_residual_interaction import (
    LSTMResidualInteraction
)


class LSTMTrajectoryModel(
    LSTMResidualInteraction
):

    def __init__(
        self,
        feature_dim=9,
        step_embed_dim=32,
        hidden_dim=64,
        lstm_layers=2,
        dropout=0.1,
        interaction_dim=10,
        future_steps=20,
    ):
        super().__init__(
            feature_dim=feature_dim,
            step_embed_dim=step_embed_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
            interaction_dim=interaction_dim,
        )

        self.future_steps = (
            future_steps
        )

        # Same trajectory head structure
        # as Final Mamba2 trajectory model
        self.trajectory_head = nn.Sequential(
            nn.Linear(
                64,
                128
            ),
            nn.LayerNorm(
                128
            ),
            nn.SiLU(),
            nn.Linear(
                128,
                future_steps * 2
            ),
        )


    def forward(
        self,
        agents,
        agent_mask,
        entry_line,
        interaction,
    ):

        B, A, T, Fdim = (
            agents.shape
        )

        # ====================================================
        # LSTM temporal branch
        # ====================================================

        x = (
            agents
            / self.feature_scale
        )

        x = x.reshape(
            B * A,
            T,
            Fdim
        )

        x = self.step_encoder(
            x
        )

        _, (h_n, _) = self.lstm(
            x
        )

        agent_embedding = (
            h_n[-1]
            .reshape(
                B,
                A,
                -1
            )
        )


        # ====================================================
        # Agent pooling
        # ====================================================

        mask = (
            agent_mask
            .unsqueeze(-1)
        )

        mean_embedding = (
            agent_embedding
            * mask
        ).sum(dim=1)

        mean_embedding = (
            mean_embedding
            /
            mask.sum(
                dim=1
            ).clamp_min(1.0)
        )

        max_embedding = (
            agent_embedding
            .masked_fill(
                mask == 0,
                -1e9
            )
            .max(
                dim=1
            )
            .values
        )

        ego_embedding = (
            agent_embedding[
                :,
                0
            ]
        )


        # ====================================================
        # Entry geometry
        # ====================================================

        entry_embedding = (
            self.entry_encoder(
                entry_line.reshape(
                    B,
                    4
                )
                / 30.0
            )
        )


        # ====================================================
        # Base scene representation
        # ====================================================

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


        # ====================================================
        # Residual interaction
        # ====================================================

        interaction_correction = (
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
            alpha
            * interaction_correction
        )


        # ====================================================
        # Original prediction heads
        # ====================================================

        decision_logits = (
            self.decision_head(
                scene
            )
        )

        time_to_entry = F.softplus(
            self.tte_head(
                scene
            )
        ).squeeze(-1)

        entry_speed = F.softplus(
            self.speed_head(
                scene
            )
        ).squeeze(-1)

        entry_heading = (
            self.heading_head(
                scene
            )
        )


        # ====================================================
        # Future trajectory head
        # ====================================================

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

            "interaction_alpha":
                alpha,

            "future_trajectory":
                future_trajectory,
        }
