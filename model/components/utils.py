import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


def smooth_rectangular_function(x: torch.Tensor, method: str = "sigmoid_based", **kwargs) -> torch.Tensor:
    """
    Computes a function that is close to 1 for x values inside the interval [0, 1] and close to 0 for values outside
    the interval. It acts as a differentiable gate function.

    Possible methods:
        - "exp_based":
            f(x) = exp( - (2x - 1) ^ 2d )
            kwargs are d > 0, integer.
        - "sigmoid_based":
            f(x) = sigmoid(alpha x) (1 - sigmoid(alpha (x-1))
            kwargs are alpha > 0.

    You can play with https://www.desmos.com/calculator entering the expressions evaluate
    the effect of the hyper params (d, alpha, etc.).
    """
    if method == "exp_based":
        d = kwargs.get("d", 1)
        return torch.exp(-(2*x - 1)**(2 * d))
    elif method == "sigmoid_based":
        alpha = kwargs.get("alpha", 1)
        return torch.sigmoid(alpha*x) * (1 - torch.sigmoid(alpha * (x-1)))
    else:
        raise NotImplementedError(f"Unknown smooth rectangular func method: '{method}'")


def add_positional_sinusoidal_encoding(x: torch.Tensor, N=10000):
    # cf. https://arxiv.org/abs/1706.03762 (Attention Is All You Need)
    #  PE(pos, 2i) = sin{ pos * (10000)^(-2i/dmodel) }
    #  PE(pos, 2i + 1) = cos{ pos * (10000)^(-2i/dmodel) }

    B, L, D = x.shape

    assert D % 2 == 0, "Positional encoding requires even D"

    i = torch.arange(0, D // 2, device=x.device, dtype=x.dtype)
    div_term = torch.exp(-np.log(N) * (2 * i / D))  # for stability: e^(-ln(N)*2i/D) = 1/e^ln(N^2i/D) = 1/N^(2i/D)
    position = torch.arange(L, device=x.device, dtype=x.dtype).unsqueeze(1)

    pe = torch.zeros_like(x)  # (B, L, D)
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)

    return x + pe


def MLP(sizes, act=nn.ReLU, final_act=None):
    """
    Multi-Layer Perceptron
    i.e. sequence of linear layers (with biases)
    After each linear layer, apply the activation `act` apart from the last.
    Apply final activation if not None.
    """
    layers = []
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            layers.append(act())
        elif final_act is not None:
            layers.append(final_act())
    return nn.Sequential(*layers)


def build_adjacency_matrix(
        N: int, edges: List[Tuple[int, int]],
) -> torch.Tensor:
    A = torch.zeros(N, N)
    for e in edges:
        i,j = tuple(e)
        A[i, j] = 1
        A[j, i] = 1
    return A


def normalize_adjacency(A: torch.Tensor, eps=1e-8):
    """
    A: (N, N) adjacency without self-loops (0/1)
    returns A_norm: (N, N) = D^{-1/2} (A + I) D^{-1/2}
    (see https://people.orie.cornell.edu/dpw/orie6334/Fall2016/lecture7.pdf)
    """
    A = A.to(torch.float32)
    N = A.shape[0]
    A_hat = A + torch.eye(N, device=A.device, dtype=A.dtype)
    deg = torch.sum(A_hat, dim=1)  # (N,)
    deg_inv_sqrt = (deg + eps).pow(-0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt  # (N,N)


class ResTConv(nn.Module):
    """
    Residual Temporal Convolution block.
    """
    def __init__(self, c, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(c, c, kernel_size, padding=pad, dilation=dilation)
        self.norm = nn.LayerNorm(c)
        self.activation = nn.GELU()

    def forward(self, X: torch.Tensor):
        """
        Input X is of shape (B, T, N, C)
        Apply convolution on temporal dimension (so X is reshaped as (B*N, C, T)
        before applying convolution because nn.Conv1d applies on (B, C_in, L_in)
        tensors C_out convolution weights tensors W_c of shape (C_in, K)).
        """
        B, T, N, C = X.shape
        X_resized = X.permute(0, 2, 3, 1).contiguous().view(B*N, C, T)  # shape=(B*N, C, T)
        out = self.conv(X_resized)  # apply convolution
        out = out.view(B, N, C, T).permute(0, 3, 1, 2).contiguous()  # reshape as (B, T, N, C)
        out = (out + X).view(B*T*N, C)  # add residual and reshape for normalization
        out = self.norm(out).view(B, T, N, C)  # apply normalization (on last dimension only) and reshape
        return self.activation(out)  # shape=(B, T, N, C)


class GCNLayer(nn.Module):
    """
    Graph Convolution layer.
    """
    def __init__(self, c_in, c_out, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out, bias=bias)
        self.activation = nn.ReLU()

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor):
        """
        Input X is of shape (B, T, N, C)
        A_norm is the normalized adjacency matrix of the graph of shape (N, N)
        (A_norm = D^-1/2 A D^-1/2
        with D=diag(d), d(i) being graph's node i degree)
        """
        A_norm = A_norm.to(X.device)

        B, T, N, C = X.shape

        X_lin = self.linear(X)  # (B, T, N, c_out)

        # batch matrix-matrix product | bmm(input, mat)_i = input_i @ mat_i
        X_lin_resized = X_lin.view(B*T, N, -1)  # (B*T, N, N)
        A_norm_resized = A_norm.unsqueeze(0).expand(B*T, -1, -1)  # (B*T, N, N)
        out = torch.bmm(A_norm_resized, X_lin_resized)
        out = out.view(B, T, N, -1)  # (B, T, N, c_out)
        return self.activation(out)  # shape=(B, T, N, c_out)


class SimpleAttentionPooling(nn.Module):
    def __init__(self, dim_in: int, attn_pool_hidden: int, attn_pool_dropout: float = 0.0):
        super().__init__()
        self.attn_score = nn.Sequential(
            nn.Linear(dim_in, attn_pool_hidden),
            nn.ReLU(),
            nn.Dropout(attn_pool_dropout),
            nn.Linear(attn_pool_hidden, 1),
        )

    def forward(self, h: torch.Tensor):
        """
        Expects h of shape (B, T, N, dim_in).
        """
        scores = self.attn_score(h)  # (B, T, N, 1)
        weights = torch.softmax(scores, dim=2)  # (B, T, N, 1) | w_k[i,j,n] = e^score[i,j,n] / sum_k e^score[i,j,k]
        out = (weights * h).sum(dim=2)  # (B, T, dim_in)
        return out


class MAB(nn.Module):
    """
    Multi(-head) Attention Block as described in J. Lee et al. "Set Transformer" (ICML 2019)
    (cf. https://arxiv.org/abs/1810.00825):

                    MAB = LayerNorm(H + rFF(H))
                    with H = LayerNorm(Q + MultiHead(Q, X, X))

    Inputs:
    - X: shape (B, L, d) (ex: skeletal motion embeddings, sentence tokens embeddings, etc.)
    - Q: shape (B, K, d) | queries
    - key_padding_mask: (B, L) (True at pad positions in X, False elsewhere)

    Note:
        the choices of the letters for the notation do not match the paper but were made to match
        our future usage in CLIP (for mean attention pooling from learned queries). So we must be careful not
        to be confused when looking at the paper (our Q = their X, our X = their Y).
    """
    def __init__(self, dim_emb: int, attn_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim_emb,
            num_heads=attn_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.layer_norm1 = nn.LayerNorm(dim_emb)
        # N.B: the paper says
        # "rFF is any row-wise feedforward layer (i.e., it processes each instance independently and identically)"
        # ==> possible rFF block below
        dim_hidden = 4 * dim_emb
        self.rff = nn.Sequential(
            nn.Linear(dim_emb, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_emb),
            nn.Dropout(dropout),
        )
        self.layer_norm2 = nn.LayerNorm(dim_emb)

    def forward(self, X: torch.Tensor,  Q: torch.Tensor, key_padding_mask: torch.Tensor):
        H, _ = self.mha(query=Q, key=X, value=X, key_padding_mask=key_padding_mask)  # (B, K, d) | returns (Output, Attention Weights)
        H = self.layer_norm1(Q + H)  # shape=(B, k, d)
        rff_H = self.rff(H)
        out = self.layer_norm2(H + rff_H)
        return out  # (B, L, d)


class DiffTimeEmbedding(nn.Module):
    """
    Diffusion time step embedding for usage in the DDPM.
    Performs MLP(Sinusoidal(t)).

    Based on the following snippets of code:
    https://github.com/haoheliu/AudioLDM/blob/main/audioldm/latent_diffusion/util.py#L173-L197
    https://github.com/haoheliu/AudioLDM/blob/main/audioldm/latent_diffusion/openaimodel.py#L524-L528
    """
    def __init__(self, num_channels: int, time_embed_dim: int = None):
        super().__init__()
        if time_embed_dim is None:
            time_embed_dim = num_channels * 4  # design choice (same as in AudioLDM repo)
        # N.B: output dim is not channels because projection occurs in the ResBlock
        self.num_channels = num_channels
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),  # output dim = time_embed_dim
        )  # (B, num_channels * 4)

    @staticmethod
    def _sinusoidal_embed(t: torch.Tensor, dim: int, max_period: int = 10000):
        """
        t: (timesteps) 1D tensor shape=(B,) i.e. one timestep per sample in the batch
        dim: output dimension
        max_period: controls the minimum frequency of the embeddings
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
        args = t[:, None]* freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb  # shape=(B, dim)

    def forward(self, t: torch.Tensor):
        return self.time_embed(self._sinusoidal_embed(t=t, dim=self.num_channels))


class GroupNorm32(nn.GroupNorm):
    """
    Special group normalization with float32 casting (with x.float()) as in:
    - https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html
    - https://github.com/haoheliu/AudioLDM/blob/main/audioldm/latent_diffusion/util.py#L240C1-L242C56
    This is for numerical stability under mixed-precision training.
    The output is cast back to the original dtype to preserve the memory and performance benefits of fp16.
    """
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ResBlock(nn.Module):
    """
    Residual conv block w/ conditioning (for U-Net).
    Chosen architecture:
        X -> GroupNorm -> Activation (SiLU) -> 1D Conv -> "+" conditioning
          -> GroupNorm -> Activation (SiLU) -> 1D Conv -> + residual (X)
    (remark: GroupNorm used because it's better than BatchNorm for ResNet + it's used in AudioLDM
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            cond_dim: int,  # dimension (channels) of the conditioning embedding
            use_scale_shift_norm: bool = False,  # FiLM (see https://arxiv.org/abs/1709.07871)
            use_sequential_condition: bool = False,  # whether the input condition is of the form (B, cond_dim, T_like)
    ):
        super().__init__()

        num_groups = 8  # 32 in Stable Diffusion # design choice (can be changed)
        kernel_size = 3  # design choice (can be changed)
        # ensure same length after conv for stride=1 and chosen kernel_side
        # (cf. formula at https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
        padding = 1

        self.norm1 = GroupNorm32(num_groups, in_channels)  # make sure: out_channels/num_groups=0
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = GroupNorm32(num_groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)

        self.use_scale_shift_norm = use_scale_shift_norm
        # conditioning projection (with optional FiLM)
        out_dim = 2 * out_channels if use_scale_shift_norm else out_channels
        if use_sequential_condition:
            self.cond_proj = nn.Conv1d(cond_dim, out_dim, kernel_size=1)  # same as Linear but no need for permutation
        else:
            self.cond_proj = nn.Linear(cond_dim, out_dim)  # if FiLM, we predict scale and shift (gamma and beta)

        # (typically to deal with up sample stage in U-Net, when in_channels = 2 * d_lat and out_channels = d_lat)
        # cf. https://github.com/haoheliu/AudioLDM/blob/main/audioldm/latent_diffusion/openaimodel.py#L246C1-L253C81
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.activation = nn.SiLU()

    def forward(self, X: torch.Tensor, cond: torch.Tensor):
        """
        Warning: X is expected as (B, d, T) -not (B, T, d)!- here because
        we apply Conv1d on the temporal dimension (T stays intact during the process).
        """
        h = self.conv1(self.activation(self.norm1(X)))  # norm -> act -> conv

        # ---- conditioning ----
        cond_out = self.cond_proj(cond)  # shape=(B, out_channels[,T_like]) or (B, 2 * out_channels[,T_like]) (if FiLM)
        if self.use_scale_shift_norm:
            scale, shift = cond_out.chunk(2, dim=1)  # split in 2 tensors of shape (B, out_channels)
            # as in AudioLDM, norm. before FiLM
            # (cf. https://github.com/haoheliu/AudioLDM/blob/main/audioldm/latent_diffusion/openaimodel.py#L278-L282)
            if cond.dim() == 2:
                h = self.norm2(h) * (1 + scale[:, :, None]) + shift[:, :, None]  # scale and shift must be (B, out_channels, 1)
            else:  # (B, out_channels, T_like) case
                h = self.norm2(h) * (1 + scale) + shift
        else:
            if cond.dim() == 2:
                h = self.norm2(h + cond_out[:, :, None])  # cond_out shape must allow broadcasting (B, out_channels, 1)
            else:  # (B, out_channels, T_like) case
                h = self.norm2(h + cond_out)
        # -----------------------

        h = self.conv2(self.activation(h))

        return self.skip_connection(X) + h


class Upsample(nn.Module):
    """
    Adapted from https://github.com/haoheliu/AudioLDM/blob/main/audioldm/latent_diffusion/openaimodel.py#L92C1-L122C17
    But only for 1D up sampling, i.e. in our case to double time dimension.
    (To be used in the U-Net.)
    Optionally also applies a convolution (without changing shape).

    Input x is expected of the shape (B, d, T').
    Output is of the shape (B, d, 2 * T')
    """
    def __init__(self, in_channels=None, out_channels=None, use_conv=False):
        super().__init__()
        self.use_conv = use_conv
        if self.use_conv:
            assert in_channels is not None, "You must provide num_channels (d) for convolution"
            out_channels = in_channels if out_channels is None else out_channels
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, X: torch.Tensor):
        """
        X of shape (B, d, T')
        Output of shape (B, d, 2 * T')
        """
        x_up = F.interpolate(X, scale_factor=2, mode="nearest")
        if self.use_conv:
            x_up = self.conv(x_up)
        return x_up  # (B, d, 2*T')
