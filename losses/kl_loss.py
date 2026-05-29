import torch
import torch.nn as nn


class KLDivergence(nn.Module):
    """
    Kullback-Leibler divergence loss.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """
    def __init__(
            self, scaling_factor: float = 1.,
            sub_kl_dims: list[int] = None,  # dims to subdivide mu and logvar last dim in multiple mus, logvars
            sub_kl_factors: list[float] = None,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor

        self.sub_kl_dims = sub_kl_dims
        if sub_kl_dims is not None:
            sub_kl_factors = sub_kl_factors if sub_kl_factors is not None else len(sub_kl_dims) * [1]
            assert len(sub_kl_dims) == len(sub_kl_factors), "num. sub. KL scaling factors must equal num. dims"
        self.sub_kl_factors = sub_kl_factors

    @staticmethod
    def compute_kl(mu, logvar):
        return 0.5 * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - logvar - 1.0)

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor):
        # mu shape=(B, T, lat_dim) or (B, lat_dim)
        # logvar shape=(B, T, lat_dim) or (B, lat_dim)

        if self.sub_kl_dims is not None:
            assert sum(self.sub_kl_dims) == mu.shape[-1] == logvar.shape[-1], "sub. dims must sum to dim_lat"

            mus = torch.split(mu, self.sub_kl_dims, dim=-1)
            logvars = torch.split(logvar, self.sub_kl_dims, dim=-1)

            out = 0.
            for sca, sub_mu, sub_logvar in zip(self.sub_kl_factors, mus, logvars):
                out += sca * self.compute_kl(sub_mu, sub_logvar)

        else:
            out = self.compute_kl(mu, logvar)

        return self.scaling_factor * out
