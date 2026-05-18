import json
import torch
import torch.nn as nn


def write_model_info(path: str, model: dict[str, nn.Module]):
    with open(path, "w") as f:
        for name, module in model.items():

            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

            f.write(f"---------- {name} ----------\n")
            f.write(f"Total parameters: {total_params}\n")
            f.write(f"Trainable parameters: {trainable_params}\n\n")

    print(f"Model info written to {path}")

def write_epoch_logs(path: str, epoch: int, stats: dict, suffix: str = ""):
    with open(path, "a") as f:
        f.write(json.dumps({"epoch": epoch, **stats}) + suffix + "\n")

def save_ckpt(
        path: str,
        epoch: int,
        model_state_dict,
        optimizer_state_dict,
        val_loss: float,
        extra_states: dict[str, dict] = None,
):
    ckpt_dict = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "val_loss": val_loss,
    }

    if extra_states:
        for k, v in extra_states.items():
            ckpt_dict[k]=v

    torch.save(ckpt_dict, path)

    print(f"Saved model and optimizer state dicts to {path}")
    if extra_states:
        print("Also saved extra state(s) dict(s) for: ", list(extra_states.keys()))

def load_ckpt(path: str, device: str = "cpu"):
    return torch.load(path, map_location=device)
