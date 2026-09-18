import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lstm_baseline import LSTMBaseline


class LSTMInteraction(LSTMBaseline):

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

        self.interaction_encoder = nn.Sequential(
            nn.Linear(interaction_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),

            nn.Linear(32, 32),
            nn.SiLU()
        )

        # v1 scene embedding = 64
        # interaction embedding = 32
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
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
        # Same temporal branch as v1 LSTM
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
            agent_embedding.masked_fill(
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

        base_scene = torch.cat(
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
                base_scene
            )
        )

        # --------------------------------------------------
        # Explicit conflict interaction branch
        # --------------------------------------------------

        interaction_embedding = (
            self.interaction_encoder(
                interaction
            )
        )

        scene = self.fusion(
            torch.cat(
                [
                    base_scene,
                    interaction_embedding
                ],
                dim=-1
            )
        )

        # --------------------------------------------------
        # Same targets as v1
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
        }
