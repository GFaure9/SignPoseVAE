import torch
import torch.nn as nn


class BodyFaceLatentTempDec(nn.Module):
    def __init__(
            self,
            T: int,
            dim_in: int = 256,
            Npts_body: int = 50,
            Npts_face: int = 128,
            dim_ft_body_dec: int = 32,
            dim_ft_face_dec: int = 16,
            n_tconv_blocks: int = 2,
            use_attn: bool = True,
            attn_heads: int = 4,
    ):
        super().__init__()
        self.T = T
        self.use_attn = use_attn
        if use_attn:
            self.mha =  nn.MultiheadAttention(embed_dim=dim_in, num_heads=attn_heads, batch_first=True)
        self.TConvBlocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim_in, dim_in, kernel_size=3, padding=1),
                nn.GELU()
            ) for _ in range(n_tconv_blocks)
        ])
        dim_out = None
        if dim_ft_body_dec and dim_ft_face_dec:
            dim_out = Npts_body * dim_ft_body_dec + Npts_face * dim_ft_face_dec
        self.to_out = None if not dim_out else nn.Linear(dim_in, dim_out)

    def forward(self, X: torch.Tensor):
        # X has the shape (B, T, dim_in)
        out = X
        if self.use_attn:
            attn_out, _ = self.mha(query=X, key=X, value=X)
            out = attn_out + out  #
        # we reshape `out` to apply convolutions on temporal dimension
        out_resized = out.permute(0, 2, 1).contiguous()  # (B, dim_in, T)
        for tconv in self.TConvBlocks:
            out_resized = tconv(out_resized)  # (B, dim_in, T)
        # get back to init shape and optionally apply projection
        out = out_resized.permute(0, 2, 1).contiguous()
        if self.to_out:  # could be eventually None if dim_in is already Npts_body*dim_ft_body + Npts_face*dim_ft_face
            out = self.to_out(out)  # to ensure correct dimension for splitter
        return out  # either (B, T, dim_in) or (B, T, dim_out)


class BodyFaceSplitter(nn.Module):
    def __init__(
            self,
            Npts_body: int = 50,
            Npts_face: int = 128,
            dim_ft_body: int = 32,
            dim_ft_face: int = 16,
    ):
        super().__init__()
        self.Npts_body = Npts_body
        self.Npts_face = Npts_face
        self.dim_ft_body = dim_ft_body
        self.dim_ft_face = dim_ft_face
        self.expected_dim = Npts_body * dim_ft_body + Npts_face * dim_ft_face

    def forward(self, X: torch.Tensor):
        B, T, C = X.shape
        assert C == self.expected_dim, "Last dimension of input tensor should be Npts_body * dim_ft_body + Npts_face * dim_ft_face"
        body = X[..., :self.Npts_body*self.dim_ft_body].view(B, T, self.Npts_body, self.dim_ft_body)  # (B, T, Npts_body, dim_ft_body)
        face = X[..., self.Npts_body*self.dim_ft_body:].view(B, T, self.Npts_face, self.dim_ft_face)  # (B, T, Npts_face, dim_ft_face)
        return body, face
