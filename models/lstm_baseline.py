import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBaseline(nn.Module):
    """
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
        step_embed_dim=32,
        hidden_dim=64,
        lstm_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        # Simple physical scaling for first baseline.
        # [x,y,vx,vy,ax,ay,sin,cos,presence]
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

        # ----------------------------------------------
        # Per-frame feature encoder
        # ----------------------------------------------
        self.step_encoder = nn.Sequential(
            nn.Linear(
                feature_dim,
                step_embed_dim
            ),
            nn.LayerNorm(
                step_embed_dim
            ),
            nn.SiLU()
        )

        # ----------------------------------------------
        # Shared temporal encoder
        # Same LSTM is used for Ego and every neighbor.
        # ----------------------------------------------
        self.lstm = nn.LSTM(
            input_size=step_embed_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ----------------------------------------------
        # Entry geometry encoder
        # ----------------------------------------------
        self.entry_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU()
        )

        # Ego embedding
        # + mean agent embedding
        # + max agent embedding
        # + map embedding
        scene_dim = (
            hidden_dim * 3
            + 32
        )

        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.SiLU()
        )

        # ----------------------------------------------
        # Multi-task heads
        # ----------------------------------------------
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

        # ----------------------------------------------
        # normalize physical input
        # ----------------------------------------------
        x = agents / self.feature_scale

        # [B,A,T,F] -> [B*A,T,F]
        x = x.reshape(
            B * A,
            T,
            Fdim
        )

        x = self.step_encoder(x)

        # Shared LSTM
        _, (h_n, _) = self.lstm(x)

        # Last LSTM layer
        agent_embedding = h_n[-1]

        agent_embedding = agent_embedding.reshape(
            B,
            A,
            -1
        )

        # ----------------------------------------------
        # Agent pooling
        # ----------------------------------------------
        mask = agent_mask.unsqueeze(-1)

        # Mean pooling
        mean_embedding = (
            agent_embedding * mask
        ).sum(dim=1)

        denom = mask.sum(
            dim=1
        ).clamp_min(1.0)

        mean_embedding = (
            mean_embedding / denom
        )

        # Max pooling
        max_input = agent_embedding.masked_fill(
            mask == 0,
            -1e9
        )

        max_embedding = (
            max_input.max(dim=1).values
        )

        # Ego is always agent 0
        ego_embedding = (
            agent_embedding[:, 0]
        )

        # ----------------------------------------------
        # Entry geometry
        # ----------------------------------------------
        entry_flat = (
            entry_line.reshape(B, 4)
            / 30.0
        )

        entry_embedding = (
            self.entry_encoder(
                entry_flat
            )
        )

        # ----------------------------------------------
        # Scene representation
        # ----------------------------------------------
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

        # ----------------------------------------------
        # Heads
        # ----------------------------------------------
        decision_logits = (
            self.decision_head(scene)
        )

        # physical quantities >= 0
        time_to_entry = F.softplus(
            self.tte_head(scene)
        ).squeeze(-1)

        entry_speed = F.softplus(
            self.speed_head(scene)
        ).squeeze(-1)

        entry_heading = (
            self.heading_head(scene)
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
        }
