"""
Original code from https://github.com/haoheliu/AudioLDM/blob/main/audioldm/latent_diffusion/attention.py

Then slightly simplified/adapted:
    - removed flash attention parts
    - removed other types of attention
    - removed checkpoints related parts
    - added context masking (to later mask pad tokens) in cross-attention
    - added SpatialTransformer1d to handle batch of inputs x of shape (B, T, d)
    - added some comments + partially adapted docs
"""


from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn
# from einops import rearrange
from entmax import entmax15  # from https://github.com/deep-spin/entmax ('sparser' softmax)


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# def Normalize(in_channels):
#     return torch.nn.GroupNorm(
#         num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
#     )


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer
    This falls-back to self-attention when conditional embeddings are not specified.
    """

    def __init__(
            self,
            query_dim,
            context_dim=None,
            heads=8,
            dim_head=64,
            dropout=0.0,
            is_inplace: bool = True,
            use_entmax: bool = False,
    ):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super().__init__()

        self.is_inplace = is_inplace
        self.use_entmax = use_entmax  # whether to use entmax instead of softmax
        self.n_heads = heads
        self.d_head = dim_head

        # Attention scaling factor
        self.scale = dim_head ** -0.5

        # The normal self-attention layer
        if context_dim is None:
            context_dim = query_dim

        # Query, key and value mappings
        d_attn = dim_head * heads
        self.to_q = nn.Linear(query_dim, d_attn, bias=False)
        self.to_k = nn.Linear(context_dim, d_attn, bias=False)
        self.to_v = nn.Linear(context_dim, d_attn, bias=False)

        # Final linear layer
        self.to_out = nn.Sequential(nn.Linear(d_attn, query_dim), nn.Dropout(dropout))

    def forward(
            self,
            x, context=None,
            context_mask=None,
            add_temporal_bias: bool = False,
            temporal_bias_alpha: float = 0.01,
            return_attn_weights: bool = False,
    ):
        """
        :param x: are the input embeddings of shape `[batch_size, T, d_model]`
        :param context: is the conditional embeddings of shape `[batch_size, L, d_cond]`
        :param context_mask: padding mask to apply on context `[batch_size, L]` (True when PAD)
        """

        # If `context` (conditioning) is `None` we perform self attention
        has_cond = context is not None
        if not has_cond:
            context = x

        # Get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Build attention mask
        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask[:, None, None, :].to(q.dtype) * -1e9  # shape=(B, 1, 1, T) or (B, 1, 1, L)

        return self.normal_attention(
            q, k, v,
            mask=attn_mask,
            add_temporal_bias=add_temporal_bias, temporal_bias_alpha=temporal_bias_alpha,
            return_attn_weights=return_attn_weights,
        )

    def normal_attention(
            self,
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            mask: torch.Tensor,
            add_temporal_bias: bool = False,
            temporal_bias_alpha: float = 0.01,
            return_attn_weights: bool = False,
    ):
        """
        #### Normal Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, T, d_attn]`
        :param k: are the key vectors before splitting heads, of shape `[batch_size, L, d_attn]`
        :param v: are the value vectors before splitting heads, of shape `[batch_size, L, d_attn]`
        :param mask: of shape `[batch_size, L]` with -inf (or -1e9) at positions to mask
        :add_temporal_bias: boolean. Whether to add temporal bias in softmax (to penalize distant frames attention).
        :temporal_bias_alpha: float. weight of the added temporal bias (if True). Default is 0.01.
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1)  # [B, T, n_heads, d_head]
        k = k.view(*k.shape[:2], self.n_heads, -1)  # [B, L, n_heads, d_head]
        v = v.view(*v.shape[:2], self.n_heads, -1)  # [B, L, n_heads, d_head]

        # Calculate attention $\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.einsum("bihd,bjhd->bhij", q, k) * self.scale  # shape=(B, n_heads, T, L)

        # Apply additive masking: Q K^T / sqrt{d} + MASK where MASK = -1e9 where to mask (0 elsewhere)
        # since later softmax( . ) will (quasi-)zero the ~ -1e9 (~ -inf) values
        # (see for instance as ref equations 2 and 3 in:
        #  Cheng et al. 2021
        #  Masked-attention Mask Transformer for Universal Image Segmentation
        #  https://arxiv.org/abs/2112.01527 )
        if mask is not None:
            attn = attn + mask  # auto broadcast over batch/head/query (since mash shape is (B, 1, 1, L))

        T, L = q.shape[1], k.shape[1]
        if add_temporal_bias and T == L:
            t = torch.arange(T, device=attn.device).view(T, 1)
            l = torch.arange(L, device=attn.device).view(1, L)
            bias = -temporal_bias_alpha * ((t - l) / T) ** 2  # [T,L]
            attn = attn + bias

        # Compute softmax
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$$
        if self.is_inplace:
            half = attn.shape[0] // 2

            if self.use_entmax:
                attn[half:] = entmax15(attn[half:], dim=-1)
                attn[:half] = entmax15(attn[:half], dim=-1)
            else:
                attn[half:] = attn[half:].softmax(dim=-1)
                attn[:half] = attn[:half].softmax(dim=-1)
        else:
            if self.use_entmax:
                attn = entmax15(attn, dim=-1)
            else:
                attn = attn.softmax(dim=-1)
        extra_output = attn.mean(dim=1) if return_attn_weights else None  # shape=(B, T, L) (averaged overs heads)

        # Compute attention output
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        out = torch.einsum("bhij,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, height * width, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)

        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out), extra_output


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        use_entmax=False,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            use_entmax=use_entmax,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(
            self,
            x,
            context=None,
            x_mask=None, context_mask=None,
            gate=None,
            scalar_gate=None,
            add_temporal_bias: bool = False, temporal_bias_alpha: float = 0.01,
            apply_cross_attention: bool = True,
            return_extra_outputs: bool = False,
    ):
        if scalar_gate is None:
            scalar_gate = 1

        # 1) Self-attention
        # ==========================
        if x_mask is not None:  # mask x tokens that should not be attended to in self-attention (typically pad tok.)
            x = scalar_gate * self.attn1(
                self.norm1(x), context=None, context_mask=x_mask,
                add_temporal_bias=add_temporal_bias, temporal_bias_alpha=temporal_bias_alpha,
                )[0] + x  # self-att + residual
        else:
            x = scalar_gate * self.attn1(
                self.norm1(x), context=None,
                add_temporal_bias=add_temporal_bias, temporal_bias_alpha=temporal_bias_alpha,
                )[0] + x  # self-att + residual / no masking (only in loss later)

        # 2) Cross-attention
        # ==========================
        cross_attn_weights = None  # when computed, of shape (B, T, L)
        if apply_cross_attention:
            cross_attn_out, cross_attn_weights = self.attn2(
                self.norm2(x),
                context=context, context_mask=context_mask,
                add_temporal_bias=add_temporal_bias, temporal_bias_alpha=temporal_bias_alpha,
                return_attn_weights=return_extra_outputs,
            )  # return output and cross-attn map (weights) if `return_attn_weights`=True
            # factor = scalar_gate if gate is None else scalar_gate * (1 + gate)
            factor = scalar_gate if gate is None else scalar_gate * gate  # to apply "gating"
            x = factor * cross_attn_out + x  # gated cross-att + residual

        # 3) Feed-forward network
        # ==========================
        x = self.ff(self.norm3(x)) + x

        # (optional) Extra outputs
        # ************************************
        extra_outputs = None
        if return_extra_outputs:
            B = x.shape[0]
            extra_outputs = [
                {
                    "cross_attention_weights": None if cross_attn_weights is None else cross_attn_weights[i],
                    "scalar_gate": scalar_gate,
                    "gate": None if gate is None else gate[i],
                }
                for i in range(B)
            ]
        # ************************************

        return x, extra_outputs


# class SpatialTransformer(nn.Module):
#     """
#     Transformer block for image-like data.
#     First, project the input (aka embedding)
#     and reshape to b, t, d.
#     Then apply standard transformer action.
#     Finally, reshape to image
#     """
#
#     def __init__(
#         self,
#         in_channels,
#         n_heads,
#         d_head,
#         depth=1,
#         dropout=0.0,
#         context_dim: int = None,
#         no_context: bool = False,
#     ):
#         super().__init__()
#
#         if no_context:
#             context_dim = None
#
#         self.in_channels = in_channels
#         inner_dim = n_heads * d_head
#         self.norm = Normalize(in_channels)
#
#         self.proj_in = nn.Conv2d(
#             in_channels, inner_dim, kernel_size=1, stride=1, padding=0
#         )
#
#         self.transformer_blocks = nn.ModuleList(
#             [
#                 BasicTransformerBlock(
#                     inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
#                 )
#                 for d in range(depth)
#             ]
#         )
#
#         self.proj_out = zero_module(
#             nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
#         )
#
#     def forward(self, x, context=None):
#         # note: if no context is given, cross-attention defaults to self-attention
#         b, c, h, w = x.shape
#         x_in = x
#         x = self.norm(x)
#         x = self.proj_in(x)
#         x = rearrange(x, "b c h w -> b (h w) c")
#         for block in self.transformer_blocks:
#             x = block(x, context=context)
#         x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
#         x = self.proj_out(x)
#         return x + x_in


class SpatialTransformer1d(nn.Module):
    """
    Transformer block for sequence-like (temporal) data.
    Input: (B, T, C)        | in our case typically sequence of latent skeletal poses
    Context: (B, L, d)      | in our case typically sequence of text tokens features (BERT, CLIP before agg., etc.)
    """
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: int = None,
        no_context: bool = False,
        use_entmax: bool = False,
        # proj_gate_attn: bool = False,  # to apply a mlp to project gate input before applying sigmoid
    ):
        super().__init__()

        if no_context:
            context_dim = None

        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        # IMPORTANT: using LayerNorm (on last dimension) instead of GroupNorm
        #            to ensure per "token"/time step normalization (not mixing time steps)
        self.norm = nn.LayerNorm(in_channels)

        # IMPORTANT: using Linear instead of Conv2d
        self.proj_in = nn.Linear(in_channels, inner_dim)
        # --- gate projection layer (optional)
        self.proj_gate_in = None
        # if proj_gate_attn:
        #     self.proj_gate_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    use_entmax=use_entmax,
                )
                for _ in range(depth)
            ]
        )

        self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))

    def forward(
            self,
            x,
            context=None,
            context_mask=None, x_mask=None,
            gate_input=None,  # expected in (B, T, 1) | with T depending on down/up level
            scalar_gate_input=None,  # expected in (B, 1)
            add_temporal_bias: bool = False, temporal_bias_alpha: float = 0.01,
            apply_cross_attention: bool = True,
            return_extra_outputs: bool = False,
    ):
        """
        x: (B, T, C)
        context: (B, L, d) or None
        context_mask: (B, L) or None (typically padding mask)
        x_mask: (B, T) or None (typically padding mask)
        Cf. https://en.wikipedia.org/wiki/Latent_diffusion_model pseudocode
        """
        x_in = x

        # normalize per token
        x = self.norm(x)

        # project to transformer dim
        x = self.proj_in(x)  # (B, T, inner_dim)

        # compute gate (optional)
        if gate_input is not None:
            g_in = gate_input.clone()
            if self.proj_gate_in is not None:
                g_in = self.proj_gate_in(gate_input)
            gate = torch.sigmoid(g_in)  # make it in [0, 1]
        else:
            gate = None

        # compute scalar gate (optional)
        if scalar_gate_input is not None:
            scalar_gate = torch.sigmoid(scalar_gate_input)[:, :, None]  # -> [0, 1] | shape=(B, 1, 1)
        else:
            scalar_gate = None

        # transformer blocks
        extra_outputs = None
        for block_num, block in enumerate(self.transformer_blocks):
            _return_extra_outputs = ((block_num==0) and return_extra_outputs)  # only saving extra outputs of 1st block
            x, extra_outputs = block(
                x,
                context=context,
                context_mask=context_mask, x_mask=x_mask,
                gate=gate,
                scalar_gate=scalar_gate,
                add_temporal_bias=add_temporal_bias, temporal_bias_alpha=temporal_bias_alpha,
                apply_cross_attention=apply_cross_attention,
                return_extra_outputs=_return_extra_outputs,
            )

        # project back
        x = self.proj_out(x)  # (B, T, C)

        return x + x_in, extra_outputs
