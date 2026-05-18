import torch
import torch.nn as nn

from .utils import GCNLayer, ResTConv, MLP


class BodyEncSpatial(nn.Module):
    def __init__(
            self,
            Npts_body: int = 50,  # number of body (i.e. torso+arms+hands) key-points (landmarks)
            dim_out: int = 64,
            dim_hidden: int = 64,
            use_gcn: bool = True,
            n_gcn_layers: int = 2,
            A_norm: torch.Tensor = None,  # shape=(Npts_body, Npts_body),
    ):
        super().__init__()
        self.Npts_body = Npts_body

        self.A_norm = None
        self.GCNs = None
        self.mlp = None

        if use_gcn:
            assert A_norm is not None, "Normalized skeletal graph adjacency required"
            self.A_norm = A_norm
            layers = []
            cur_lay_last_dim = 3  # (x, y, z)
            for i in range(n_gcn_layers):
                next_lay_last_dim = dim_out if i == n_gcn_layers - 1 else max(cur_lay_last_dim, dim_out)
                layers.append(GCNLayer(c_in=cur_lay_last_dim, c_out=next_lay_last_dim))
                cur_lay_last_dim = next_lay_last_dim
            self.GCNs = nn.ModuleList(layers)
        else:
            self.mlp = MLP(sizes=[3, dim_hidden, dim_out], act=nn.ReLU, final_act=None)

    def forward(self, X: torch.Tensor):
        if self.GCNs is not None:
            out = X  # input shape=(B, T, Npts_body, 3)
            A_norm = self.A_norm.to(device=X.device)
            for gcn in self.GCNs:
                out = gcn(X=out, A_norm=A_norm)
        else:
            B, T, N, C = X.shape
            X_resized = X.reshape(B * T * N, C)  # MLP applied on last dim
            out = self.mlp(X_resized)
            out = out.view(B, T, N, -1)  # last dimension is dim_out
        return out  # shape=(B, T, Npts_body, dim_out)


class BodyEncTemp(nn.Module):
    def __init__(
            self,
            Npts_body: int = 50,
            dim_in: int = 64,
            dim_out: int = 128,
            n_restconv_blocks: int = 2,
            convt_kernel_size: int = 3,
            convt_dilation: int = 1,
            use_attn: bool = True,
            attn_heads: int = 4,
    ):
        super().__init__()
        self.Npts_body = Npts_body
        self.ResTConvBlocks = nn.ModuleList(
            [ResTConv(
                c=dim_in,
                kernel_size=convt_kernel_size,
                dilation=convt_dilation,
            ) for _ in range(n_restconv_blocks)]
        )
        self.pool_joints = lambda h: h.mean(
            dim=2)  # average pooling over body joints --> resulting shape=(B, T, dim_in)
        self.use_attn = use_attn  # whether to use MHA before last projection layer
        if self.use_attn:
            self.mha = nn.MultiheadAttention(embed_dim=dim_in, num_heads=attn_heads, batch_first=True)
        self.proj = nn.Linear(dim_in, dim_out)  # with bias

    def forward(self, X: torch.Tensor):
        h = X
        for restconv in self.ResTConvBlocks:
            h = restconv(X=h)  # (B, T, Npts_body, dim_in)
        h_pooled = self.pool_joints(h)
        if self.use_attn:  # self-attention
            attn_out, _ = self.mha(query=h_pooled, key=h_pooled, value=h_pooled)  # shape=(B, T, dim_in)
            h_pooled = h_pooled + attn_out  # adding residual (classical attention block)
        out = self.proj(h_pooled)  # (B, T, dim_out)
        return out  # shape=(B, T, dim_out)


class BodyDecSpatial(nn.Module):
    def __init__(
            self,
            dim_in: int = 32,
            dim_hidden: int = 64,
            Npts_body: int = 50,
    ):
        super().__init__()
        self.Npts_body = Npts_body
        self.mlp = MLP(sizes=[dim_in, dim_hidden, 3], act=nn.ReLU, final_act=None)

    def forward(self, X: torch.Tensor):
        B, T, N, C = X.shape  # C=dim_in
        out = X.reshape(B * T * N, C)
        out = self.mlp(out).view(B, T, N, 3)  # MLP applied on last dim | shape=(B, T, Npts_body, 3)
        return out  # shape=(B, T, Npts_body, 3)
