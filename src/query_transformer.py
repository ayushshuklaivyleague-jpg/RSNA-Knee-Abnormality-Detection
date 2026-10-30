"""
src/query_transformer.py
========================
Finding-Query Multi-Instance Cross-Attention Network for RSNA Knee MRI.
12 learned finding queries attend across 2D slices with plane & depth embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]


class FindingQueryMIAttention(nn.Module):
    """
    12 learned finding queries query a pool of multi-plane slice tokens.
    """

    def __init__(self, dim: int = 384, num_findings: int = 12, num_planes: int = 3):
        super().__init__()
        self.num_findings = num_findings
        self.dim = dim

        # 12 learned finding queries
        self.finding_queries = nn.Parameter(torch.randn(1, num_findings, dim) * 0.02)

        # Physical plane (Sagittal=0, Coronal=1, Axial=2) and z-depth embeddings
        self.plane_emb = nn.Embedding(num_planes, dim)
        self.depth_proj = nn.Linear(1, dim)

        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # Finding-specific classification MLPs
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            ) for _ in range(num_findings)
        ])

    def forward(self, slice_tokens: torch.Tensor, plane_ids: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slice_tokens: [B, N_slices, D]
            plane_ids:    [B, N_slices] (0, 1, 2)
            depths:       [B, N_slices, 1] normalized depth in [0, 1]
        Returns:
            logits: [B, 12]
        """
        B = slice_tokens.shape[0]

        # Inject physical plane and depth embeddings into slice tokens
        kv = slice_tokens + self.plane_emb(plane_ids) + self.depth_proj(depths)
        kv = self.norm_kv(kv)

        # Expand finding queries across batch
        q = self.finding_queries.expand(B, -1, -1)
        q = self.norm_q(q)

        # Multi-head cross attention
        attended, _ = self.cross_attn(query=q, key=kv, value=kv)

        # Predict finding logits
        logits = [self.classifiers[i](attended[:, i, :]) for i in range(self.num_findings)]
        return torch.cat(logits, dim=-1)
