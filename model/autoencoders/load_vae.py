from .abstract_vae import VAE
from .skelmotionvae import SkelMotionVAE
from .skelmotionmultivae import SkelMotionMultiRegionVAE


implemented_vaes = [
    "SkelMotionVAE",
    "SkelMotionMultiRegionVAE",
]

def load_vae(name: str, cfg) -> VAE:
    if name == "SkelMotionVAE":
        return SkelMotionVAE(cfg)
    elif name == "SkelMotionMultiRegionVAE":
        return SkelMotionMultiRegionVAE(cfg)
    else:
        raise NotImplementedError(
            f"'{name}' is not a valid VAE class. "
            f"Please choose among: {implemented_vaes}"
        )
