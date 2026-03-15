"""
EfficientNet backbone for BirdCLEF+ 2026 sound event detection.

Wraps timm EfficientNet models with a SED (Sound Event Detection) head
that produces per-species logits from mel spectrogram input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Attention-based pooling over time dimension for SED."""

    def __init__(self, in_features: int):
        super().__init__()
        self.attention = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, time, features) -> (batch, features)"""
        attn_weights = torch.softmax(self.attention(x), dim=1)
        return (x * attn_weights).sum(dim=1)


class EfficientNetSED(nn.Module):
    """EfficientNet backbone with SED head for multi-label classification.

    Input: mel spectrogram (batch, 1, n_mels, time_frames)
    Output: logits (batch, n_classes)

    Args:
        model_name: timm model name (e.g., 'tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns').
        n_classes: Number of target species.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate before final classifier.
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnet_b0_ns",
        n_classes: int = 206,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        import timm

        # Create backbone, removing default classifier
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,  # mono spectrogram
            num_classes=0,
            global_pool="",  # we do our own pooling
        )
        self.feature_dim = self.backbone.num_features

        # SED head: attention pooling + classifier
        self.attention_pool = AttentionPool(self.feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel spectrogram, shape (batch, 1, n_mels, time_frames).

        Returns:
            Logits, shape (batch, n_classes).
        """
        # Extract features: (batch, features, h, w)
        features = self.backbone(x)

        # Reshape to (batch, time, features) for attention pooling
        b, c, h, w = features.shape
        features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # Attention pool over time, then classify
        pooled = self.attention_pool(features)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities for submission."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
