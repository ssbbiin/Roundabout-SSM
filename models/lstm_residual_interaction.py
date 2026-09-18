import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lstm_baseline import LSTMBaseline


class LSTMResidualInteraction(LSTMBaseline):

    def __init__(
        self,
        feature_dim=9,
        step_embed_dim=32,
        hidden_dim=64,
        lstm_layers=2,
        dropout=0.1,
        interaction_dim=10,
    ):
        super().__init__(
            feature_dim=feature_dim,
            step_embed_dim=step_embed_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )

        # 10-d interaction → 64-d correction
        self.interaction_encoder = nn.Sequential(
            nn.Linear(interaction_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),

            nn.Linear(32, 64),
        )

        # Start EXACTLY from v1 behavior.
        self.interaction_alpha = nn.Parameter(
            torch.tensor(0.0)
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
        # v1 trajectory branch
        # --------------------------------------------------

        x = agents / self.feature_scale

        x = x.reshape(
            B * A,
            T,
            Fdim
        )

        x = self.step_encoder(x)

        _, (h_n, _) = self.lstm(x)

        agent_embedding = (
            h_n[-1]
            .reshape(B, A, -1)
        )

        mask = agent_mask.unsqueeze(-1)

        mean_embedding = (
            agent_embedding * mask
        ).sum(dim=1)

        mean_embedding = (
            mean_embedding
            /
            mask.sum(dim=1).clamp_min(1.0)
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
        # Static entry geometry
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

        # This is exactly the v1 64-d scene representation.
        base_scene = (
            self.scene_encoder(
                scene_input
            )
        )

        # --------------------------------------------------
        # Residual interaction correction
        # --------------------------------------------------

        interaction_correction = (
            self.interaction_encoder(
                interaction
            )
        )

        # tanh keeps the scalar bounded.
        alpha = torch.tanh(
            self.interaction_alpha
        )

        scene = (
            base_scene
            +
            alpha
            * interaction_correction
        )

        # --------------------------------------------------
        # Original v1 heads
        # --------------------------------------------------

        return {
            "decision_logits":
                self.decision_head(scene),

            "time_to_entry":
                F.softplus(
                    self.tte_head(scene)
                ).squeeze(-1),

            "entry_speed":
                F.softplus(
                    self.speed_head(scene)
                ).squeeze(-1),

            "entry_heading":
                self.heading_head(scene),

            "interaction_alpha":
                alpha,
        }
