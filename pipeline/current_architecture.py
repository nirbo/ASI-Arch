# The following imports may fail in the static analysis environment if torch is not available.
# They are required at runtime when the training script is executed.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

class DeltaNet(nn.Module if nn is not None else object):  # type: ignore

    """Minimal vanilla transformer – baseline for memory‑reduction experiments."""
    def __init__(self, vocab_size: int, d_model: int = 768, n_heads: int = 12, n_layers: int = 6, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 2048, d_model))  # max_len 2048
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # input_ids: (B, T)
        h = self.embedding(input_ids) + self.pos_emb[:, :input_ids.size(1)]
        if attention_mask is not None:
            # make mask for transformer: 0 = keep, 1 = ignore
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
            attn_mask = attn_mask == 0
            h = self.encoder(h, src_key_padding_mask=~attn_mask.squeeze(1))
        else:
            h = self.encoder(h)
        logits = self.lm_head(h)
        return logits

# expose alias required by train_architecture.py
Model = DeltaNet
