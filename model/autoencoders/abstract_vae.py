import torch
import torch.nn as nn
from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple


class VAE(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def encoder_modules_names(self):
        pass

    @property
    @abstractmethod
    def decoder_modules_names(self):
        pass

    @abstractmethod
    def encode(self, X: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def decode(self, Z: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def get_encoder_modules(self):
        return nn.ModuleList([getattr(self, name) for name in self.encoder_modules_names])

    def freeze_decoder(self):
        for name in self.decoder_modules_names:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def unfreeze_decoder(self):
        for name in self.decoder_modules_names:
            for param in getattr(self, name).parameters():
                param.requires_grad = True

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # (B, T, dim_latent) [or (B, dim_latent)]

    def forward(self, X: torch.Tensor, decode_mu: bool = False, pad_mask=None):
        mu, logvar, z = self.encode(X, pad_mask=pad_mask)
        if decode_mu:
            X_recon = self.decode(mu, pad_mask=pad_mask)
        else:
            X_recon = self.decode(z, pad_mask=pad_mask)
        return X_recon, mu, logvar, z
