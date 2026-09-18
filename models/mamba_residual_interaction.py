import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba_baseline import MambaBaseline


class MambaResidualInteraction(MambaBaseline):

    def __init__(
        self,
        feature_dim=9,
        d_model=64,
        d_state=64,
        num_layers=2,
        dropout=0.1,
        interaction_dim=10,
    ):
        super().__init__(
            feature_dim=feature_dim,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.interaction_encoder = nn.Sequential(
            nn.Linear(interaction_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 64),
        )

        # alpha=0 → 시작 시점은 정확히 Mamba v1
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
        # v1 Mamba trajectory branch
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

        # 정확히 기존 Mamba v1 scene
        base_scene = (
            self.scene_encoder(
                scene_input
            )
        )

        # --------------------------------------------------
        # Residual interaction correction
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
            + alpha * correction
        )

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
