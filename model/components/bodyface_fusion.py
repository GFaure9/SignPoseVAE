import torch
import torch.nn as nn


class BodyFaceFusion(nn.Module):
    def __init__(
            self,
            dim_in_body: int = 128,
            dim_in_face: int = 128,
            dim_out: int = 256,
            use_attn: bool = True,
            attn_heads: int = 4,
    ):
        super().__init__()
        self.use_attn = use_attn
        dim_in = dim_in_body + dim_in_face
        if use_attn:
            self.mha = nn.MultiheadAttention(embed_dim=dim_in, num_heads=attn_heads, batch_first=True)
        self.proj_act = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU()
        )

    def forward(self, emb_body: torch.Tensor, emb_face: torch.Tensor):
        cat_emb = torch.cat([emb_body, emb_face], dim=-1)  # shape=(B, T, dim_body+dim_face)
        if self.use_attn:
            attn_out, _ = self.mha(query=cat_emb, key=cat_emb, value=cat_emb)
            cat_emb = cat_emb + attn_out  # shape=(B, T, dim_body+dim_face)
        # cat_emb = cat_emb.mean(dim=1)  # temporal average pooling (B, dim_body+dim_face)  # uncomment for (B, dim_out)
        out = self.proj_act(cat_emb)  # (B, T, dim_out) [or (B, dim_out) if previous line is uncommented]
        return out
