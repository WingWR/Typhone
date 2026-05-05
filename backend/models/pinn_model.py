from __future__ import annotations

import torch
from torch import nn

FEATURE_COLUMNS = ["t_hours", "lng", "lat", "wind_speed", "pressure"]
STATE_COLUMNS = ["lng", "lat", "wind_speed", "pressure"]
PINN_OUTPUT_COLUMNS = ["lng", "lat", "wind_speed", "pressure", "u_mps", "v_mps"]


class TyphoonPINN(nn.Module):
    """MLP PINN that predicts next typhoon state plus local east/north velocity."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(PINN_OUTPUT_COLUMNS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.network(features)
        state = torch.tanh(raw[:, : len(STATE_COLUMNS)])
        velocity = 80.0 * torch.tanh(raw[:, len(STATE_COLUMNS) :])
        return torch.cat([state, velocity], dim=1)
