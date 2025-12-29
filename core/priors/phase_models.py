"""
Phase-wise transformer classifiers (pretrained).
"""

import torch
import torch.nn as nn
import joblib
from pathlib import Path


class SimpleTransformerClassifier(nn.Module):
    """
    Transformer classifier used for phase-wise TTP prediction.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).mean(dim=1)
        return self.fc(x)


def load_phase_model(
    phase_name: str,
    input_dim: int,
    model_root: Path,
    device: torch.device,
):
    """
    Load pretrained transformer + label encoder for a phase.
    """
    phase_dir = model_root / phase_name

    label_encoder = joblib.load(phase_dir / "label_encoder.pkl")

    model = SimpleTransformerClassifier(
        input_dim=input_dim,
        num_classes=len(label_encoder.classes_),
    )

    model.load_state_dict(torch.load(phase_dir / "model.pt", map_location=device))

    model.to(device)
    model.eval()

    return model, label_encoder
