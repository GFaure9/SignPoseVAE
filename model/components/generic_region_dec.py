import torch
import torch.nn as nn
from typing import List

from .utils import MLP, ResTConv


class GenericSkelRegionMotionDecoder(nn.Module):
    def __init__(
            self,
            Npts: int,  # number of points in the region
            dim_in: int,
            hidden_dims: List[int] = None,  # as many as hidden layers in MLP
            # --- Temporal conv params
            use_temp_conv: bool = False,  # whether to use Residual Temp Conv
            n_restconv_blocks: int = None,
            convt_kernel_size: int = None,
            convt_dilation: int = None,
    ):
        """
        Input tensor will be of the form (B, T, Npts, dim_in) - x,y,z coordinates.
        `dim_in` typically matches corresponding region encoder dimension (for symmetry).
        """
        super().__init__()
        self.Npts = Npts

        self.CrossAttn = None
        self.ResTConvBlocks = None

        if use_temp_conv:
            self.ResTConvBlocks = nn.ModuleList(
                [ResTConv(
                    c=dim_in,
                    kernel_size=convt_kernel_size,
                    dilation=convt_dilation,
                ) for _ in range(n_restconv_blocks)]
            )

        self.mlp = MLP(sizes=[dim_in, *hidden_dims, 3], act=nn.ReLU, final_act=None)

    def forward(self, Z: torch.Tensor):

        # ======/ Temporal decoding (optional)
        if self.ResTConvBlocks is not None:
            for restconv in self.ResTConvBlocks:
                Z = restconv(X=Z)  # (B, T, Npts, dim_in)

        # ======/ Spatial decoding
        B, T, N, C = Z.shape
        Z_resized = Z.reshape(B * T * N, C)  # MLP applied on last dim
        out = self.mlp(Z_resized)
        out = out.view(B, T, N, -1)  # last dimension is 3

        return out  # shape=(B, T, Npts, 3)
