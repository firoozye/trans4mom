import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleBiasLSTM(nn.Module):
    """
    LSTM implementation with single bias vector per gate (4 * hidden_size),
    strictly matching Keras / TensorFlow implementation in Wood et al. (2022).
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias)
        # Forget gate bias initialized to 1.0
        with torch.no_grad():
            self.bias[self.hidden_size : 2 * self.hidden_size].fill_(1.0)

    def forward(
        self, 
        x: torch.Tensor, 
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        if hx is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h_t, c_t = hx

        x_ih = F.linear(x, self.weight_ih) # [B, T, 4 * H]
        outputs = []

        for t in range(seq_len):
            gates = x_ih[:, t, :] + F.linear(h_t, self.weight_hh, self.bias)
            i, f, g, o = gates.chunk(4, dim=-1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)
            outputs.append(h_t)

        output_seq = torch.stack(outputs, dim=1)
        return output_seq, (h_t, c_t)


class GateAddNorm(nn.Module):
    """
    Gated Linear Unit (GLU) + Residual Addition + Layer Normalization (epsilon = 1e-3).
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, eps: float = 0.001):
        super().__init__()
        self.glu_dense = nn.Linear(input_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        glu_val = self.glu_dense(x)
        val, gate = glu_val.chunk(2, dim=-1)
        gated = val * torch.sigmoid(gate)
        gated = self.dropout(gated)
        if residual is not None:
            return self.layer_norm(gated + residual)
        return self.layer_norm(gated + x)


class GatedResidualNetwork(nn.Module):
    """
    Standard Gated Residual Network (GRN) from Lim et al. (2019) / Wood et al. (2022).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
        eps: float = 0.001
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        self.dense_1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.context_dense = nn.Linear(context_dim, hidden_dim, bias=False)

        self.dense_2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate_add_norm = GateAddNorm(hidden_dim, output_dim, dropout=dropout, eps=eps)

        if input_dim != output_dim:
            self.skip_dense = nn.Linear(input_dim, output_dim)
        else:
            self.skip_dense = None

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.dense_1(x)
        if context is not None:
            if context.dim() == 2 and h.dim() == 3:
                h = h + self.context_dense(context).unsqueeze(1)
            else:
                h = h + self.context_dense(context)
        h = F.elu(h)
        h = self.dense_2(h)

        residual = self.skip_dense(x) if self.skip_dense is not None else x
        return self.gate_add_norm(h, residual=residual)


class MaskedInterpretableMultiHeadAttention(nn.Module):
    """
    Masked Interpretable Multi-Head Attention (MIMHA) with shared value projection.
    Lim et al. (2019) / Wood et al. (2022) Section IV.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_head: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = 1.0 / (d_head ** 0.5)

        # 4 independent query and key projections
        self.q_heads = nn.ModuleList([nn.Linear(d_model, d_head, bias=False) for _ in range(n_heads)])
        self.k_heads = nn.ModuleList([nn.Linear(d_model, d_head, bias=False) for _ in range(n_heads)])

        # Single shared value projection across all heads
        self.v_shared = nn.Linear(d_model, d_head, bias=False)

        # Final linear projection back to d_model
        self.out_proj = nn.Linear(d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        v = self.v_shared(x) # [B, T, d_head]

        head_outputs = []
        head_weights = []

        # Causal mask (strictly lower triangular, preventing lookahead)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        for q_proj, k_proj in zip(self.q_heads, self.k_heads):
            q = q_proj(x) # [B, T, d_head]
            k = k_proj(x) # [B, T, d_head]

            scores = torch.bmm(q, k.transpose(1, 2)) * self.scale # [B, T, T]
            scores = scores.masked_fill(causal_mask, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            weights = self.dropout(weights)

            head_out = torch.bmm(weights, v) # [B, T, d_head]
            head_outputs.append(head_out)
            head_weights.append(weights)

        # Average head outputs across heads: (1 / H) * sum(head_out)
        avg_head_out = torch.stack(head_outputs, dim=0).mean(dim=0) # [B, T, d_head]
        output = self.out_proj(avg_head_out) # [B, T, d_model]
        avg_weights = torch.stack(head_weights, dim=0).mean(dim=0) # [B, T, T]

        return output, avg_weights
