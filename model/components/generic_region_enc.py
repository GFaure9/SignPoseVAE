import torch
import torch.nn as nn
from typing import List

from .utils import MLP, ResTConv, GCNLayer, SimpleAttentionPooling


class GenericSkelRegionMotionEncoder(nn.Module):
    def __init__(
            self,
            Npts: int,  # number of points in the region
            dim_out: int,
            hidden_dims: List[int] = None,  # as many as hidden layers in MLP
            # --- GCN params
            use_gcn: bool = False,  # whether to use Graph Convolutions
            n_gcn_layers: int = None,
            A_norm=None,
            # --- Temporal conv params
            use_temp_conv: bool = False,  # whether to use Residual Temp Conv
            n_restconv_blocks: int = None,
            convt_kernel_size: int = None,
            convt_dilation: int = None,
            # --- Attention pooling
            use_attn_pooling: bool = False,  # whether to use attn pooling instead of mean pooling over joints dim
    ):
        """
        Input tensor will be of the form (B, T, Npts, 3) - x,y,z coordinates.
        """
        super().__init__()
        self.Npts = Npts

        self.A_norm = None
        self.GCNs = None
        self.ResTConvBlocks = None
        self.mlp = None
        self.attn_pool = None

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
            self.mlp = MLP(sizes=[3, *hidden_dims, dim_out], act=nn.ReLU, final_act=None)

        if use_temp_conv:
            self.ResTConvBlocks = nn.ModuleList(
                [ResTConv(
                    c=dim_out,
                    kernel_size=convt_kernel_size,
                    dilation=convt_dilation,
                ) for _ in range(n_restconv_blocks)]
            )

        if use_attn_pooling:
            self.attn_pool = SimpleAttentionPooling(
                dim_in=dim_out,
                attn_pool_hidden=dim_out // 2,  # design choice for simplicity (can be changed)
                attn_pool_dropout=0.0,
            )

    def forward(self, X: torch.Tensor):

        # ======/ Spatial encoding

        # > SKEL. GRAPH CONVOLUTION
        if self.GCNs is not None:
            h = X  # input shape=(B, T, Npts, 3)
            A_norm = self.A_norm.to(device=X.device)
            for gcn in self.GCNs:
                h = gcn(X=h, A_norm=A_norm)

        # > MLP
        else:
            B, T, N, C = X.shape
            X_resized = X.reshape(B * T * N, C)  # MLP applied on last dim
            h = self.mlp(X_resized)
            h = h.view(B, T, N, -1)  # last dimension is dim_out

        # ======/ Temporal encoding (optional)
        if self.ResTConvBlocks is not None:
            for restconv in self.ResTConvBlocks:
                h = restconv(X=h)  # (B, T, Npts, dim_out)

        # ======/ Attention pooling (optional -> else normal mean pooling)
        if self.attn_pool is not None:
            out = self.attn_pool(h)  # (B, T, dim_out)
        else:
            out = h.mean(dim=2)  # average pooling over points --> resulting shape=(B, T, dim_out)

        return out  # shape=(B, T, dim_out)
