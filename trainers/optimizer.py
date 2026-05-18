from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler


def get_optimizer(trainable_params, name: str = "adam", **kwargs) -> Optimizer:
    if name == "adam":
        optim_params = {
            "lr": kwargs.get("lr", 1e-3),
            "betas": tuple(kwargs.get("betas", [0.9, 0.999])),
            "eps": kwargs.get("eps", 1e-8),
            "weight_decay": kwargs.get("weight_decay", 0.0)
        }
        return Adam(params=trainable_params, **optim_params)
    else:
        raise ValueError(f"'{name}' is not a valid optimizer name")


def get_scheduler(optimizer: Optimizer, name: str = "plateau", **kwargs):
    if name == "plateau":
        schedule_params = {
            "factor": kwargs.get("factor", 0.5),  # decreasing factor (multiply lr by it when plateau)
            "patience": kwargs.get("patience", 5),  # number of evals without improvement to be considered a plateau
            "threshold": kwargs.get("threshold", 1e-4),  # new_loss<best_loss*(1−threshold)? yes => improvement
            # "verbose": kwargs.get("verbose", False),
        }
        # mode = 'min' because we want to minimize validation loss so improvement is when inferior loss
        return ReduceLROnPlateau(optimizer, mode='min', **schedule_params)
    elif name == "cosine_annealing":
        schedule_params = {
            "T_max": kwargs.get("T_max", 1000),  # max number of epochs
            "eta_min": kwargs.get("eta_min", 1e-5),  # lr min that will be reached
        }
        return CosineAnnealingLR(optimizer, **schedule_params)
    elif name == "cosine_annealing_warm_restarts":
        # cf. https://docs.pytorch.org/docs/2.11/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        schedule_params = {
            "T_0": kwargs.get("T_0", 50),  #  num iterations until the 1st restart
            "T_mult": kwargs.get("T_mult", 2),  # factor multiplying T_i (num it. before restart) after a restart
            "eta_min": kwargs.get("eta_min", 1e-5),  # lr min that will be reached
        }
        return CosineAnnealingWarmRestarts(optimizer, **schedule_params)
    else:
        raise ValueError(f"'{name}' is not a valid scheduler name")


def get_scaler(name: str = "grad", **kwargs):
    if name == "grad":
        return GradScaler(enabled=True)
    else:
        raise ValueError(f"'{name}' is not a valid scaler name")
