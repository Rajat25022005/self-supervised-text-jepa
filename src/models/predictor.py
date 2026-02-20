import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    The Predictor bridges Student latents → predicted Teacher latents.

    This is the heart of JEPA. Given the student's representation of the
    *visible* (unmasked) context, the predictor must guess what the teacher
    would produce for the *masked* positions.

    Key design decision — positional conditioning:
    ─────────────────────────────────────────────
    The predictor needs to know *where* in the sequence it is predicting.
    Without positional information, the predictor sees identical input for
    every masked token and can only output a single averaged representation,
    regardless of which span it's supposed to predict.

    We inject this by adding a learned positional embedding (indexed by token
    position 0..max_len-1) to the latent before the MLP layers. This way,
    predicting position 42 vs position 87 produces different outputs even
    given identical context — which is what we want.

    Architecture:
        context_latent (B, T, proj_dim)
            ↓  + positional embedding
        (B, T, proj_dim)
            ↓
        Linear(proj_dim → predictor_hidden_dim)
            ↓  GELU
        Linear(predictor_hidden_dim → predictor_hidden_dim)
            ↓  GELU
        Linear(predictor_hidden_dim → proj_dim)
            ↓
        predicted target latents (B, T, proj_dim)
    """

    def __init__(self, proj_dim: int, hidden_dim: int, max_len: int = 128):
        """
        Args:
            proj_dim   : dimension of the latent space (must match encoder proj_dim)
            hidden_dim : width of the MLP hidden layers
            max_len    : maximum sequence length (for positional embeddings)
        """
        super().__init__()

        # Learned positional embeddings — one vector per position
        # These are added to the latent, not concatenated, to keep dim constant
        self.pos_emb = nn.Embedding(max_len, proj_dim)

        # 3-layer MLP with bottleneck expansion
        self.net = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim, bias=False),
        )

    def forward(
        self,
        z: torch.Tensor,            # (B, T, proj_dim) — student latents
        mask_positions: torch.Tensor,  # (B, T) bool — which positions to predict
    ) -> torch.Tensor:
        """
        Args:
            z             : student encoder output (B, T, proj_dim)
            mask_positions: boolean tensor marking masked token positions

        Returns:
            pred : (N, proj_dim) — predictions for masked positions only
                   where N = total number of masked tokens in the batch
        """
        B, T, D = z.shape
        device = z.device

        # Build position indices: [[0,1,2,...,T-1], [0,1,...,T-1], ...]
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B, T)

        # Add positional conditioning to every token's latent
        z_pos = z + self.pos_emb(positions)  # (B, T, D)

        # Pass through MLP
        z_pred = self.net(z_pos)  # (B, T, D)

        # Extract only the masked positions — shape (N, D)
        # This is what we compute loss on: predicted vs teacher's masked latents
        pred_masked = z_pred[mask_positions]  # (N, D)

        return pred_masked