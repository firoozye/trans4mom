import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from models.layers import (
    SingleBiasLSTM, 
    GateAddNorm, 
    GatedResidualNetwork, 
    MaskedInterpretableMultiHeadAttention
)

class WoodMomentumTransformer(nn.Module):
    """
    Exact, Strict Implementation of the Decoder-Only Temporal Fusion Transformer
    from Wood, Roberts & Zohren (2022) (arXiv:2112.08534v3 / J. Financial Data Science).
    
    Architectural ground truth:
    1. Dedicated continuous feature GRNs in Variable Selection Network (8 for baseline, 10 for CPD).
    2. 4 static context GRNs (c_s, c_e, c_c, c_h) initializing LSTM state.
    3. Masked Interpretable MHA with shared value weights.
    4. Direct position sizing head z_t = tanh(W h_t + b). STRICTLY NO RESIDUAL PRIOR.
    5. Hidden dimension d = 64, 4 attention heads of size 16 (380,462 params baseline, 425,016 params CPD).
    6. LayerNorm epsilon = 1e-3.
    """
    BASE_FEATURE_NAMES = [
        'norm_daily_return', 'norm_monthly_return', 'norm_quarterly_return',
        'norm_half_year_return', 'norm_annual_return',
        'macd_8_24', 'macd_16_48', 'macd_32_96'
    ]
    CPD_FEATURE_NAMES = [
        'cpd_sev_21', 'cpd_sev_126'
    ]
    FEATURE_NAMES = BASE_FEATURE_NAMES

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        d_head: int = 16,
        num_asset_classes: int = 5,
        dropout: float = 0.1,
        eps: float = 0.001,
        include_cpd: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.num_asset_classes = num_asset_classes
        self.include_cpd = include_cpd
        self.feature_names = list(self.BASE_FEATURE_NAMES)
        if include_cpd:
            self.feature_names.extend(self.CPD_FEATURE_NAMES)

        # 1. Feature 1D Linear Projections (8 or 10 features -> 64)
        self.feature_projections = nn.ModuleDict({
            feat: nn.Linear(1, d_model) for feat in self.feature_names
        })

        # 2. Static Asset Class Embedding (5 categories -> 64)
        self.asset_class_embedding = nn.Embedding(num_asset_classes, d_model)

        # 3. Static Variable Selection Network
        self.static_vsn_logits = GatedResidualNetwork(d_model, d_model, 1, dropout=dropout, eps=eps)
        self.static_vsn_asset_class = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout, eps=eps)

        # 4. Static Context Encoders (4 GRNs: c_s, c_e, c_c, c_h)
        self.static_context_var_selection = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout, eps=eps)
        self.static_context_enrichment = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout, eps=eps)
        self.static_context_state_c = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout, eps=eps)
        self.static_context_state_h = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout, eps=eps)

        # 5. Temporal Variable Selection Network: Dedicated Feature GRNs
        self.temporal_vsn_logits = GatedResidualNetwork(
            input_dim=d_model * len(self.feature_names),
            hidden_dim=d_model,
            output_dim=len(self.feature_names),
            context_dim=d_model,
            dropout=dropout,
            eps=eps
        )
        self.temporal_vsn_features = nn.ModuleDict({
            feat: GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout, eps=eps)
            for feat in self.feature_names
        })

        # 6. Temporal Local Processing: Single-Bias LSTM + GateAddNorm
        self.lstm = SingleBiasLSTM(d_model, d_model)
        self.local_gate_add_norm = GateAddNorm(d_model, d_model, dropout=dropout, eps=eps)

        # 7. Temporal Static Enrichment GRN
        self.temporal_static_enrichment = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            context_dim=d_model,
            dropout=dropout,
            eps=eps
        )

        # 8. Temporal Global Processing: Masked Interpretable MHA + GateAddNorm
        self.attention = MaskedInterpretableMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout
        )
        self.global_gate_add_norm = GateAddNorm(d_model, d_model, dropout=dropout, eps=eps)

        # 9. Position-wise Feed-Forward Decoder GRN + Final Skip
        self.decoder_positionwise = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=d_model,
            dropout=dropout,
            eps=eps
        )
        self.decoder_final_skip = GateAddNorm(d_model, d_model, dropout=dropout, eps=eps)

        # 10. Direct Output Position Sizing Head (Wood et al. 2022 Eq. 10: z_t = tanh(W h_t + b))
        self.position_sizing = nn.Linear(d_model, 1)

    def forward(
        self, 
        raw_features: Dict[str, torch.Tensor],
        asset_class_id: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        raw_features: Dict containing 8 (or 10) tensors of shape [Batch, SeqLen, 1]
        asset_class_id: [Batch] categorical integer tensor (0..4)
        """
        batch_size = asset_class_id.shape[0]

        # 1. Linear Feature Projections -> [B, T, d_model]
        projected = {
            feat: self.feature_projections[feat](raw_features[feat])
            for feat in self.feature_names
        }
        seq_len = projected[self.feature_names[0]].shape[1]

        # 2. Static Context Processing
        asset_emb = self.asset_class_embedding(asset_class_id) # [B, d_model]
        static_weight = torch.sigmoid(self.static_vsn_logits(asset_emb)) # [B, 1]
        static_feat = self.static_vsn_asset_class(asset_emb) # [B, d_model]
        static_vsn_out = static_weight * static_feat # [B, d_model]

        c_s = self.static_context_var_selection(static_vsn_out) # [B, d_model]
        c_e = self.static_context_enrichment(static_vsn_out)    # [B, d_model]
        c_c = self.static_context_state_c(static_vsn_out)       # [B, d_model]
        c_h = self.static_context_state_h(static_vsn_out)       # [B, d_model]

        # 3. Temporal Variable Selection Network
        stacked_projections = [projected[feat] for feat in self.feature_names]
        flattened_features = torch.cat(stacked_projections, dim=-1) # [B, T, num_feats * d_model]

        vsn_logits = self.temporal_vsn_logits(flattened_features, context=c_s) # [B, T, num_feats]
        vsn_weights = F.softmax(vsn_logits, dim=-1) # [B, T, num_feats]

        processed_features = [
            self.temporal_vsn_features[feat](projected[feat])
            for feat in self.feature_names
        ] # num_feats x [B, T, d_model]
        stacked_processed = torch.stack(processed_features, dim=2) # [B, T, num_feats, d_model]

        temporal_vsn_out = (vsn_weights.unsqueeze(-1) * stacked_processed).sum(dim=2) # [B, T, d_model]

        # 4. Temporal Local Processing: SingleBiasLSTM initialized with (c_h, c_c)
        lstm_out, _ = self.lstm(temporal_vsn_out, hx=(c_h, c_c)) # [B, T, d_model]
        local_out = self.local_gate_add_norm(lstm_out, residual=temporal_vsn_out) # [B, T, d_model]

        # 5. Temporal Static Enrichment
        enrich_out = self.temporal_static_enrichment(local_out, context=c_e) # [B, T, d_model]

        # 6. Temporal Global Processing: Masked Interpretable MHA
        attn_out, attn_weights = self.attention(enrich_out) # [B, T, d_model]
        global_out = self.global_gate_add_norm(attn_out, residual=enrich_out) # [B, T, d_model]

        # 7. Positionwise Decoder + Final Skip from local_out
        pw_out = self.decoder_positionwise(global_out) # [B, T, d_model]
        decoder_out = self.decoder_final_skip(pw_out, residual=local_out) # [B, T, d_model]

        # 8. Direct Position Regression (Equation 10: z_t = tanh(W h_t + b))
        positions = torch.tanh(self.position_sizing(decoder_out)) # [B, T, 1]

        if return_attention:
            return positions, attn_weights
        return positions

# Canonical Model Alias
MomentumTransformer = WoodMomentumTransformer
