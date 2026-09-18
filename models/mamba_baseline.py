import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba2


class ResidualMamba2Block(nn.Module):
    def __init__(
        self,
        d_model=64,
        d_state=64,
        d_conv=4,
        expand=2,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=d_model,

            # Current laptop-compatible path
            rmsnorm=False,
            use_mem_eff_path=False,
        )

    def forward(self, x):
        return x + self.mamba(
            self.norm(x)
        )


class MambaBaseline(nn.Module):

    def __init__(
        self,
        feature_dim=9,
        d_model=64,
        d_state=64,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        # Same physical scaling used in LSTM baseline
        feature_scale = torch.tensor(
            [
                30.0, 30.0,
                15.0, 15.0,
                10.0, 10.0,
                1.0, 1.0,
                1.0
            ],
            dtype=torch.float32
        )

        self.register_buffer(
            "feature_scale",
            feature_scale
        )

        # -----------------------------------------
        # Per-frame feature projection
        # -----------------------------------------
        self.input_encoder = nn.Sequential(
            nn.Linear(
                feature_dim,
                d_model
            ),
            nn.LayerNorm(
                d_model
            ),
            nn.SiLU()
        )

        # -----------------------------------------
        # Temporal SSM backbone
        # -----------------------------------------
        self.blocks = nn.ModuleList([
            ResidualMamba2Block(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            d_model
        )

        # -----------------------------------------
        # Entry geometry
        # Same size as LSTM version
        # -----------------------------------------
        self.entry_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU()
        )

        # Ego + mean + max + map
        scene_dim = (
            d_model * 3
            + 32
        )

        self.scene_encoder = nn.Sequential(
            nn.Linear(
                scene_dim,
                128
            ),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(
                128,
                64
            ),
            nn.SiLU()
        )

        # -----------------------------------------
        # Same four heads as LSTM
        # -----------------------------------------
        self.decision_head = nn.Linear(
            64, 2
        )

        self.tte_head = nn.Linear(
            64, 1
        )

        self.speed_head = nn.Linear(
            64, 1
        )

        self.heading_head = nn.Linear(
            64, 2
        )

    def forward(
        self,
        agents,
        agent_mask,
        entry_line,
    ):

        B, A, T, Fdim = agents.shape

        # -----------------------------------------
        # Normalize
        # -----------------------------------------
        x = (
            agents /
            self.feature_scale
        )

        # [B,A,T,F]
        # ->
        # [B*A,T,F]
        x = x.reshape(
            B * A,
            T,
            Fdim
        )

        x = self.input_encoder(x)

        # -----------------------------------------
        # Mamba2 temporal processing
        # -----------------------------------------
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # Current time step representation
        agent_embedding = x[:, -1]

        agent_embedding = (
            agent_embedding.reshape(
                B,
                A,
                -1
            )
        )

        # -----------------------------------------
        # Agent pooling
        # -----------------------------------------
        mask = (
            agent_mask.unsqueeze(-1)
        )

        mean_embedding = (
            agent_embedding * mask
        ).sum(dim=1)

        denom = (
            mask.sum(dim=1)
            .clamp_min(1.0)
        )

        mean_embedding = (
            mean_embedding /
            denom
        )

        max_input = (
            agent_embedding.masked_fill(
                mask == 0,
                -1e9
            )
        )

        max_embedding = (
            max_input.max(dim=1).values
        )

        ego_embedding = (
            agent_embedding[:, 0]
        )

        # -----------------------------------------
        # Entry geometry
        # -----------------------------------------
        entry_flat = (
            entry_line.reshape(
                B, 4
            )
            / 30.0
        )

        entry_embedding = (
            self.entry_encoder(
                entry_flat
            )
        )

        # -----------------------------------------
        # Scene fusion
        # -----------------------------------------
        scene = torch.cat(
            [
                ego_embedding,
                mean_embedding,
                max_embedding,
                entry_embedding,
            ],
            dim=-1
        )

        scene = self.scene_encoder(
            scene
        )

        # -----------------------------------------
        # Multi-task output
        # -----------------------------------------
        return {
            "decision_logits":
                self.decision_head(
                    scene
                ),

            "time_to_entry":
                F.softplus(
                    self.tte_head(
                        scene
                    )
                ).squeeze(-1),

            "entry_speed":
                F.softplus(
                    self.speed_head(
                        scene
                    )
                ).squeeze(-1),

            "entry_heading":
                self.heading_head(
                    scene
                ),
        }
