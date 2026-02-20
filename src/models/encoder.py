import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    The backbone encoder used by both Student (context) and Teacher (target).

    Architecture:
        Input tokens
            ↓
        DistilBERT (or any AutoModel) — produces (B, T, hidden_size)
            ↓
        LayerNorm — stabilizes representations before projection
            ↓
        Linear projection — maps hidden_size → proj_dim (our latent space)

    The projection head maps from the transformer's native hidden size (768 for
    DistilBERT) down to a smaller `proj_dim` (256). Working in a smaller latent
    space makes the predictor's job cleaner and is standard in SSL methods like
    BYOL and I-JEPA.

    Both the student and teacher are instances of this same class. The teacher's
    weights are a slow-moving EMA of the student's weights — they are never
    updated by backprop directly.
    """

    def __init__(self, model_name: str, proj_dim: int):
        super().__init__()

        # Load pretrained transformer backbone
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for DistilBERT

        # Projection: hidden_size → proj_dim
        # LayerNorm before linear stabilizes training significantly
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, proj_dim, bias=False),
        )

    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T)
        attention_mask: torch.Tensor,      # (B, T)
    ) -> torch.Tensor:
        """
        Returns:
            z : (B, T, proj_dim) — token-level latent representations
        """
        # Get token-level hidden states from transformer
        # last_hidden_state shape: (B, T, hidden_size)
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        # Project to latent space: (B, T, hidden_size) → (B, T, proj_dim)
        z = self.proj(out)
        return z