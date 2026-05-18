import torch
import torch.nn as nn

from .utils import MLP, ResTConv


class FaceEncSpatial(nn.Module):
    def __init__(
            self,
            Npts_face: int = 128,  # number of facial key-points (landmarks)
            dim_out: int = 48,
            dim_hidden: int = 64,
    ):
        super().__init__()
        self.Npts_face = Npts_face
        self.mlp = MLP(sizes=[3, dim_hidden, dim_out], act=nn.ReLU, final_act=None)

    def forward(self, X: torch.Tensor):
        B, T, N, C = X.shape
        X_resized = X.reshape(B*T*N, C)  # MLP applied on last dim | reshape() instead view() to handle non contiguous
        out = self.mlp(X_resized)
        out = out.view(B, T, N, -1)  # last dimension is dim_out
        return out  # shape=(B, T, Npts_face, dim_out)


class FaceEncTemp(nn.Module):
    def __init__(
            self,
            Npts_face: int = 128,
            dim_in: int = 48,
            dim_out: int = 128,
            n_restconv_blocks: int = 2,
            convt_kernel_size: int = 3,
            convt_dilation: int = 1,
            use_attn: bool = True,
            attn_heads: int = 4,
    ):
        super().__init__()
        self.Npts_face = Npts_face
        self.ResTConvBlocks = nn.ModuleList(
            [ResTConv(
                c=dim_in,
                kernel_size=convt_kernel_size,
                dilation=convt_dilation,
            ) for _ in range(n_restconv_blocks)]
        )
        self.pool_joints = lambda h: h.mean(dim=2)  # average pooling over face joints --> resulting shape=(B, T, dim_in)
        self.use_attn = use_attn  # whether to use MHA before last projection layer
        if self.use_attn:
            self.mha = nn.MultiheadAttention(embed_dim=dim_in, num_heads=attn_heads, batch_first=True)
        self.proj = nn.Linear(dim_in, dim_out)  # with bias

    def forward(self, X: torch.Tensor):
        h = X
        for restconv in self.ResTConvBlocks:
            h = restconv(X=h)  # (B, T, Npts_face, dim_in)
        h_pooled = self.pool_joints(h)
        if self.use_attn:  # self-attention
            attn_out, _ = self.mha(query=h_pooled, key=h_pooled, value=h_pooled)  # shape=(B, T, dim_in)
            h_pooled = h_pooled + attn_out  # adding residual (classical attention block)
        out = self.proj(h_pooled)  # (B, T, dim_out)
        return out  # shape=(B, T, dim_out)


class FaceDecSpatial(nn.Module):
    def __init__(
            self,
            dim_in: int = 16,
            dim_hidden: int = 32,
            Npts_face: int = 128,
    ):
        super().__init__()
        self.Npts_face = Npts_face
        self.mlp = MLP(sizes=[dim_in, dim_hidden, 3], act=nn.ReLU, final_act=None)

    def forward(self, X: torch.Tensor):
        B, T, N, C = X.shape  # C=dim_in
        out = X.reshape(B*T*N, C)
        out = self.mlp(out).view(B, T, N, 3)  # MLP applied on last dim | shape=(B, T, Npts_face, 3)
        return out  # shape=(B, T, Npts_face, 3)
