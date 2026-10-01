import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBaseline(nn.Module):
    """
    Attention-based temporal baseline.

    Input
    -----
    agents:
        [B, A, T, F]
        B = batch
        A = 9 agents (ego + 8 neighbors)
        T = 50 frames
        F = 9 features

    agent_mask:
        [B, A]

    entry_line:
        [B, 2, 2]

    Output
    ------
    decision_logits : [B, 2]
    time_to_entry   : [B]
    entry_speed     : [B]
    entry_heading   : [B, 2]
    """

    def __init__(
        self,
        feature_dim=9,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_history_frames=50,
    ):
        super().__init__()

        # Same physical scaling as LSTM / Mamba2
        feature_scale = torch.tensor(
            [
                30.0, 30.0,
                15.0, 15.0,
                10.0, 10.0,
                1.0, 1.0,
                1.0,
            ],
            dtype=torch.float32
        )

        self.register_buffer(
            "feature_scale",
            feature_scale
        )

        self.max_history_frames = max_history_frames

        # Per-frame feature projection
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

        # Learned positional embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_history_frames,
                d_model
            )
        )

        nn.init.normal_(
            self.pos_embedding,
            mean=0.0,
            std=0.02
        )

        # Transformer temporal encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(
                d_model
            )
        )

        # Same entry geometry encoder
        self.entry_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU()
        )

        # Ego + mean + max + entry geometry
        scene_dim = (
            d_model * 3
            + 32
        )

        # Same scene encoder
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

        # Same prediction heads
        self.decision_head = nn.Linear(
            64,
            2
        )

        self.tte_head = nn.Linear(
            64,
            1
        )

        self.speed_head = nn.Linear(
            64,
            1
        )

        self.heading_head = nn.Linear(
            64,
            2
        )


    def forward(
        self,
        agents,
        agent_mask,
        entry_line,
    ):
        B, A, T, Fdim = agents.shape

        if T > self.max_history_frames:
            raise ValueError(
                f"History length {T} exceeds "
                f"max_history_frames="
                f"{self.max_history_frames}"
            )

        # Physical normalization
        x = (
            agents
            / self.feature_scale
        )

        # [B,A,T,F] -> [B*A,T,F]
        x = x.reshape(
            B * A,
            T,
            Fdim
        )

        # Frame embedding
        x = self.input_encoder(
            x
        )

        # Add temporal position
        x = (
            x
            + self.pos_embedding[
                :,
                :T,
                :
            ]
        )

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(
                T,
                T,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        # Transformer
        x = self.transformer(
            x,
            mask=causal_mask
        )

        # Current time-step representation
        agent_embedding = (
            x[:, -1]
        )

        agent_embedding = (
            agent_embedding.reshape(
                B,
                A,
                -1
            )
        )

        # Agent pooling
        mask = (
            agent_mask.unsqueeze(-1)
        )

        mean_embedding = (
            agent_embedding
            * mask
        ).sum(dim=1)

        denom = (
            mask.sum(dim=1)
            .clamp_min(1.0)
        )

        mean_embedding = (
            mean_embedding
            / denom
        )

        max_input = (
            agent_embedding.masked_fill(
                mask == 0,
                -1e9
            )
        )

        max_embedding = (
            max_input
            .max(dim=1)
            .values
        )

        ego_embedding = (
            agent_embedding[:, 0]
        )

        # Entry geometry
        entry_flat = (
            entry_line.reshape(
                B,
                4
            )
            / 30.0
        )

        entry_embedding = (
            self.entry_encoder(
                entry_flat
            )
        )

        # Scene representation
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
