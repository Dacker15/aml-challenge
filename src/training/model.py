import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Pre-normalized residual block.
    """

    def __init__(self, dim, hidden_factor=4, dropout=0.5):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_factor, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class MLP(nn.Module):
    """
    Multi-layer perceptron with residual blocks.
    """

    def __init__(
        self,
        input_dim=1024,
        output_dim=1536,
        hidden_dim=1024,
        num_layers=3,
        dropout=0.5,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        x = self.output_proj(x)
        return x
