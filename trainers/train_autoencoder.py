import os
import yaml
import torch
from tqdm import tqdm
from pathlib import Path

from .optimizer import get_optimizer, get_scheduler
from ..model.autoencoders import load_vae
from ..losses import KLDivergence, ReconstructionLoss
from ..data.dataset import SLPDataset, build_slp_dataloader
from ..data.tokenize import get_tokenizer
from ..utils.logs_checkpoints import save_ckpt, load_ckpt, write_epoch_logs, write_model_info


class VAETrainer:
    def __init__(
            self,
            model,
            recon_loss_fn,
            kl_loss_fn,
            optimizer,
            device: str = "cuda",
            beta_start: float = 0.,
            beta_end: float = 1.,
            warmup_epochs: int = 10,
            recon_annealing_epochs: dict[str, int] = None,
            with_clip: bool = False,
            clip_model: LatentSignLanguageCLIP = None,
            clip_loss_fn: CLIPLoss = None,
            clip_loss_scaling: float = 1.,
            # ----- maybe for the future... --------
            # (For now VAE decoder will always be frozen when jointly training CLIP model)
            # freeze_vae_decoder_epochs: int = 0,
            # --------------------------------------
            skels_pad_value: float = 0.,
            txt_input_ids_pad_value: int = 0,

    ):
        self.device = device
        self.model = model.to(device)

        # == losses
        self.recon_loss_fn = recon_loss_fn
        self.kl_loss_fn = kl_loss_fn

        # == optimizer
        self.optimizer = optimizer

        # == KL annealing parameters
        # https://arxiv.org/abs/1804.03599 (Understanding disentangling in beta-VAE C.P. Burgess et al. (2018))
        # https://arxiv.org/abs/2310.15440 (Learning Dynamics in Linear VAE: Posterior Collapse Threshold, Superfluous Latent Space Pitfalls, and Speedup with KL Annealing - 2023)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_epochs = warmup_epochs

        # == Reconstruction loss annealing parameters
        self.alpha_start = 0.
        self.alpha_end = 1.
        self.recon_annealing_epochs = recon_annealing_epochs

        # == CLIP joint training
        if with_clip:
            assert (clip_model is not None), "For jointly train a CLIP model, a trainable CLIP model must be provided"
            assert (clip_loss_fn is not None), "For jointly train a CLIP model, a CLIP loss function must be provided"
        self.with_clip = with_clip
        self.clip_model = clip_model
        self.clip_loss_fn = clip_loss_fn
        self.clip_loss_scaling = clip_loss_scaling
        # self.freeze_vae_decoder_epochs = freeze_vae_decoder_epochs
        # -- pad values (to allow for padding mask reconstruction later)
        self.skels_pad_value = skels_pad_value
        self.txt_input_ids_pad_value = txt_input_ids_pad_value

    def compute_beta(self, epoch, method: str = "monotonic"):
        if method == "monotonic":
            if epoch >= self.warmup_epochs:
                return self.beta_end
            return self.beta_start + (self.beta_end - self.beta_start) * epoch / self.warmup_epochs
        else:
            raise ValueError(f"'{method}' is not a valid method for beta computation")

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        beta = self.compute_beta(epoch)

        loss_recon_tot = 0.
        loss_kl_tot = 0.
        loss_tot = 0.
        loss_clip_tot = 0.

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            x = batch["skels"].to(self.device)
            B, T = x.shape[:2]
            # NB: the padding mask is not always used (it depends on the VAE)
            x_pad_mask = ~(x.view(B, T, -1) != self.skels_pad_value).any(dim=-1)  # (B, T) | boolean (True=>pad)
            x_recon, mu, logvar, z = self.model(x, pad_mask=x_pad_mask)

            if self.recon_annealing_epochs:
                afs = {
                    # k: min(self.alpha_end,self.alpha_start+(self.alpha_end-self.alpha_start)*epoch/v) for k, v in self.recon_annealing_epochs.items()
                    k: int(epoch >= v) for k, v in self.recon_annealing_epochs.items()
                }
                recon_loss_val = self.recon_loss_fn(x_recon, x, annealing_factors=afs)
            else:
                recon_loss_val = self.recon_loss_fn(x_recon, x)

            kl_loss_val = self.kl_loss_fn(mu, logvar)
            loss_val = recon_loss_val + beta * kl_loss_val

            # ******* Computing CLIP model loss *******
            if self.with_clip:
                txt_input_ids = batch["text"].to(self.device)
                input_ids_pad_mask = (txt_input_ids == self.txt_input_ids_pad_value)
                z_clip_emb, txt_clip_emb = self.clip_model(
                    skelmotion_latent_X=z,
                    text_input_ids=txt_input_ids,
                    skelmotion_padding_mask=x_pad_mask,
                    input_ids_padding_mask=input_ids_pad_mask,
                )
                clip_loss_val = self.clip_loss_fn(z_clip_emb, txt_clip_emb)
                loss_val += self.clip_loss_scaling * clip_loss_val
                loss_clip_tot += clip_loss_val.item()
            # *****************************************

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            # update losses for logs
            loss_recon_tot += recon_loss_val.item()
            loss_kl_tot += kl_loss_val.item()
            loss_tot += loss_val.item()

        num_batches = len(dataloader)

        output = {
            "loss": loss_tot / num_batches,
            "recon": loss_recon_tot / num_batches,
            "kl": loss_kl_tot / num_batches,
            "beta": beta,
        }
        if self.with_clip:
            output["clip"] = loss_clip_tot / num_batches

        return output

    def run(
            self,
            train_dataloader,
            val_dataloader,
            output_dir: str,
            max_epoch: int,
            val_freq: int = 10,
            lr_min: float = 1e-6,
            scheduler=None,
            random_seed: int = 42,
    ):
        # -- set random seed (for re-parameterization trick) to ensure reproducibility
        torch.manual_seed(random_seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

        os.makedirs(output_dir, exist_ok=True)
        train_logs_path = os.path.join(output_dir, "train.log")
        val_logs_path = os.path.join(output_dir, "validation.txt")
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        last_val_ckpt_prefix = "every"
        best_val_ckpt_prefix = "best"

        train_history = []
        val_history = []
        best_val_loss = None
        best = True

        for epoch in range(1, max_epoch + 1):

            # =========================== TRAINING ===========================
            train_stats = self.train_epoch(train_dataloader, epoch)

            # -- write training logs
            train_history.append(train_stats)
            write_epoch_logs(path=train_logs_path, epoch=epoch, stats=train_stats)
            train_stats_clip_log = "" if not self.with_clip else f"clip={train_stats['clip']:.4f}  |"
            print(
                f"Epoch {epoch}:    "
                f"loss={train_stats['loss']:.4f}  |"
                f"recon={train_stats['recon']:.4f}  |"
                f"kl={train_stats['kl']:.4f}  |"
                f"beta={train_stats['beta']:.4f}  |" + train_stats_clip_log
            )

            # ============= VALIDATION (every `val_freq` epochs) =============
            if epoch % val_freq == 0:
                clip_kwargs = {
                    "clip_model": self.clip_model,
                    "clip_loss_fn": self.clip_loss_fn,
                    "skels_pad_value": self.skels_pad_value,
                    "txt_input_ids_pad_value": self.txt_input_ids_pad_value,
                }
                val_stats = evaluate(
                    model=self.model,
                    dataloader=val_dataloader,
                    recon_loss_fn=self.recon_loss_fn,
                    kl_loss_fn=self.kl_loss_fn,
                    device=self.device,
                    **clip_kwargs
                )

                # -- update best validation loss
                if best_val_loss is None:
                    best_val_loss = val_stats["loss"]
                else:
                    best = (val_stats["loss"] < best_val_loss)

                # -- saving ckpt for best validation loss so far
                if best:
                    best_val_loss = val_stats["loss"]
                    save_ckpt(
                        path=os.path.join(ckpt_dir, f"{best_val_ckpt_prefix}.ckpt"),
                        epoch=epoch,
                        model_state_dict=self.model.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        val_loss=best_val_loss,
                    )
                    # ********** Save CLIP model + optimizer (best) **********
                    if self.with_clip:
                        save_ckpt(
                            path=os.path.join(ckpt_dir, f"{best_val_ckpt_prefix}_clip.ckpt"),
                            epoch=epoch,
                            model_state_dict=self.clip_model.state_dict(),
                            optimizer_state_dict=self.optimizer.state_dict(),
                            val_loss=best_val_loss,
                            extra_states={"clip_loss_state_dict": self.clip_loss_fn.state_dict()},
                        )
                    # ********************************************************

                # -- write validation logs & save ckpt (replace last validation epoch one)
                val_history.append(val_stats)
                write_epoch_logs(path=val_logs_path, epoch=epoch, stats=val_stats, suffix=" *" if best else "")
                save_ckpt(
                    path=os.path.join(ckpt_dir, f"{last_val_ckpt_prefix}.ckpt"),
                    epoch=epoch,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    val_loss=val_stats["loss"],
                )

                # ********* Save CLIP model + optimizer (every) **********
                if self.with_clip:
                    save_ckpt(
                        path=os.path.join(ckpt_dir, f"{last_val_ckpt_prefix}_clip.ckpt"),
                        epoch=epoch,
                        model_state_dict=self.clip_model.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        val_loss=val_stats["loss"],
                        extra_states={"clip_loss_state_dict": self.clip_loss_fn.state_dict()},
                    )
                # *********************************************************

                val_stats_clip_log = "" if not self.with_clip else f"clip={val_stats['clip']:.4f}  |"
                print(
                    f"==> Validation metrics:    "
                    f"loss={val_stats['loss']:.4f}  |"
                    f"recon={val_stats['recon']:.4f}  |"
                    f"kl={val_stats['kl']:.4f}  |" + val_stats_clip_log
                )

                # -- update scheduler based on validation loss
                if scheduler is not None:
                    scheduler.step(val_stats["loss"])  # specific for scheduler based on val loss (e.g. 'plateau')

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr < lr_min:
                early_stop_log = f"Stopping early: learning rate {current_lr:.2e} < {lr_min:.2e}"
                with open(train_logs_path, "a") as f:
                    f.write(early_stop_log + "\n")
                print(early_stop_log)
                break

        terminate_log = f"Training terminated: best validation loss = {best_val_loss: .4f}"
        with open(train_logs_path, "a") as f:
            f.write(terminate_log + "\n")
        print(terminate_log)

        return train_history, val_history


@torch.no_grad()
def evaluate(
        # -- general args
        model,
        dataloader,
        recon_loss_fn,
        kl_loss_fn,
        device="cuda",
        # -- CLIP args (if clip_moel or clip_loss_fn is set to None, nothing else is done)
        clip_model=None,
        clip_loss_fn=None,
        skels_pad_value: float = None,
        txt_input_ids_pad_value: int = None,
):
    model.eval()
    loss_recon_tot = 0.
    loss_kl_tot = 0.
    loss_tot = 0.
    loss_clip_tot = 0.

    for batch in tqdm(dataloader):
        x = batch["skels"].to(device)
        B, T = x.shape[:2]
        x_pad_mask = ~(x.view(B, T, -1) != skels_pad_value).any(dim=-1)   # (B, T) | boolean
        x_recon, mu, logvar, z = model(x, decode_mu=True, pad_mask=x_pad_mask)

        recon_loss_val = recon_loss_fn(x_recon, x)
        kl_loss_val = kl_loss_fn(mu, logvar)
        loss_val = recon_loss_val + kl_loss_val

        # ******* Computing CLIP model loss *******
        if clip_model and clip_loss_fn:
            txt_input_ids = batch["text"].to(device)
            input_ids_pad_mask = (txt_input_ids == txt_input_ids_pad_value)
            B, T = x.shape[:2]
            mu_clip_emb, txt_clip_emb = clip_model(
                skelmotion_latent_X=mu,  # using mean (mu) as skeletal motion latent embedding
                text_input_ids=txt_input_ids,
                skelmotion_padding_mask=x_pad_mask,
                input_ids_padding_mask=input_ids_pad_mask,
            )
            clip_loss_val = clip_loss_fn(mu_clip_emb, txt_clip_emb)
            loss_val += clip_loss_val
            loss_clip_tot += clip_loss_val.item()
        # ******************************************

        loss_recon_tot += recon_loss_val.item()
        loss_kl_tot += kl_loss_val.item()
        loss_tot += loss_val.item()

    num_batches = len(dataloader)

    output = {
        "loss": loss_tot / num_batches,
        "recon": loss_recon_tot / num_batches,
        "kl": loss_kl_tot / num_batches,
    }
    if clip_model and clip_loss_fn:
        output["clip"] = loss_clip_tot / num_batches

    return output


def train_skelmotionvae(cfg: dict):
    # -- save config in output dir
    output_dir = cfg["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir + "/config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # -- whether to train a CLIP model (assumes first VAE training have already been performed and ckpt saved)
    clip_joint_training = cfg["training"].get("with_clip", False)

    # -- loading datasets & building dataloaders
    train_data_path = cfg["data"]["train"]["path"]
    val_data_path = cfg["data"]["dev"]["path"]
    train_path_kwarg = {f"{cfg['data']['train']['type']}_path": train_data_path}
    val_path_kwarg = {f"{cfg['data']['dev']['type']}_path": val_data_path}
    data_params = {
        "skel_field": cfg["data"]["skel_field"],
        "id_field": cfg["data"]["id_field"],
        "skip_frames": cfg["data"]["skip_frames"],
    }
    if clip_joint_training:  # to train CLIP model, we need textual inputs
        data_params["text_field"] = cfg["data"]["text_field"]
        data_params["max_sent_len"] = cfg["data"].get("max_sent_len", None)
        data_params["text_tokenizer"] = get_tokenizer(name=cfg["model"]["clip"]["text_enc"]["pretrained_model_name"])
        data_params["gloss_field"] = None

    train_data = SLPDataset(**train_path_kwarg, **data_params, load_only_skel=~clip_joint_training)
    val_data = SLPDataset(**val_path_kwarg, **data_params, load_only_skel=~clip_joint_training)

    pad_value = 0.0
    dataloader_params = {
        "fixed_seq_len": cfg["model"]["T"],  # fixed length (number of frames)
        "pad_value": pad_value,
        "batch_size": cfg["training"]["batch_size"],
    }
    train_dataloader = build_slp_dataloader(dataset=train_data, shuffle=True, **dataloader_params)
    val_dataloader = build_slp_dataloader(dataset=val_data, shuffle=False, **dataloader_params)

    # -- device
    device = cfg["training"].get("device", "cpu")

    # -- whether to continue from a given checkpoint for VAE
    continue_training = cfg["training"]["continue"]
    ckpt = None
    if continue_training:
        ckpt_path = cfg["training"]["ckpt"]
        print(f"Continuing VAE training from {ckpt_path}")
        ckpt = load_ckpt(ckpt_path, device=device)
    if clip_joint_training:
        assert continue_training and (ckpt is not None), "For CLIP training, pre-trained VAE ckpt is required"

    # -- whether to continue from a given checkpoint for CLIP
    continue_clip_training = cfg["training"].get("continue_clip_training", False)
    clip_ckpt = None
    if continue_clip_training:
        clip_ckpt_path = cfg["training"]["clip_ckpt"]
        print(f"Continuing CLIP training from {clip_ckpt_path}")
        clip_ckpt = load_ckpt(clip_ckpt_path, device=device)

    # -- building model
    skelmotionvae = load_vae(name=cfg["model"].get("name", "SkelMotionVAE"), cfg=cfg["model"])
    modules = {}
    if ckpt:
        skelmotionvae.load_state_dict(ckpt["model_state_dict"], strict=False)
    if clip_joint_training:
        skelmotionvae.freeze_decoder()  # WARNING: freezing VAE decoder for CLIP joint training
        latentslclip = LatentSignLanguageCLIP(cfg["model"]["clip"])
        if clip_ckpt:
            latentslclip.load_state_dict(clip_ckpt["model_state_dict"])
        modules["LatentSignLanguageCLIP"] = latentslclip
    modules["SkelMotionVAE"] = skelmotionvae

    # -- building losses
    reconstruction_losses_kwargs = cfg["losses"]["recon"]
    recon_loss = ReconstructionLoss(**reconstruction_losses_kwargs, batch_norm=True, target_pad=pad_value)
    kl_loss_kwargs = cfg["losses"].get("kl", {"scaling_factor": 1.})
    kl_loss = KLDivergence(**kl_loss_kwargs)
    if clip_joint_training:
        clip_loss = CLIPLoss(**cfg["losses"]["clip"], scaling_factor=1.)
        if clip_ckpt:
            clip_loss.load_state_dict(clip_ckpt["clip_loss_state_dict"])
        modules["CLIPLoss"] = clip_loss

    # -- write model info
    write_model_info(path=output_dir + "/model_info.txt", model=modules)

    # -- building optimizer & scheduler (optional)
    trainable_params = []
    for module in modules.values():
        trainable_params += list(module.parameters())
    optimizer = get_optimizer(
        trainable_params=trainable_params,
        name=cfg["training"]["optimizer"],
        **cfg["training"]["optim_params"]
    )
    # WARNING: If joint clip training, we should not start from the pre-trained VAE optimizer state.
    #          Instead, a new optimizer should be initialized, or if continue from the last training
    #          of a CLIP model, we can take the optimizer state from clip_ckpt.
    if ckpt and not clip_joint_training:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if clip_ckpt:
        optimizer.load_state_dict(clip_ckpt["optimizer_state_dict"])

    scheduler = get_scheduler(
        optimizer=optimizer,
        name=cfg["training"]["scheduler"],
        **cfg["training"]["schedule_params"]
    )

    # -- building trainer
    vae_trainer = VAETrainer(
        model=skelmotionvae,
        recon_loss_fn=recon_loss, kl_loss_fn=kl_loss,
        optimizer=optimizer,
        device=device,
        # **** beta params ****
        beta_start=cfg["training"]["beta_start"],
        beta_end=cfg["training"]["beta_end"],
        warmup_epochs=cfg["training"]["warmup_epochs"],
        # **** alpha params (optional) ****
        recon_annealing_epochs=cfg["losses"].get("recon_annealing_epochs", None),
        # **** pad values (when CLIP joint training) ****
        skels_pad_value=pad_value,
        txt_input_ids_pad_value=getattr(train_data.text_tokenizer, "pad_id", None),
    )

    # -- training model
    vae_trainer.run(
        output_dir=output_dir + "/training",
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        val_freq=cfg["training"]["validation_frequency"],
        max_epoch=cfg["training"]["max_epoch"],
        lr_min=cfg["training"]["learning_rate_min"],
        scheduler=scheduler,
    )


@torch.no_grad()
def compute_skelmotionvae_predictions(
        output_folder: str,
        cfg_model: dict,
        ckpt_path: str,
        cfg_data: dict,
        data_name: str = "test",
        device: str = "cuda",
        batch_size: int = 64,
        pad_value: float = 0.0,
):
    predictions_folder = output_folder + f"/predictions_{Path(ckpt_path).stem}"
    os.makedirs(predictions_folder, exist_ok=True)

    # -- load model w/ trained weights (checkpoint)
    ckpt = load_ckpt(ckpt_path, device=device)
    skelmotionvae = load_vae(name=cfg_model.get("name", "SkelMotionVAE"), cfg=cfg_model)
    skelmotionvae.load_state_dict(ckpt["model_state_dict"], strict=False)
    skelmotionvae = skelmotionvae.to(device=device)

    # -- build test dataset & dataloader
    data_kwargs = cfg_data[data_name]
    data_path_kwarg = {f"{data_kwargs['type']}_path": data_kwargs["path"]}
    data_params = {
        "skel_field": cfg_data["skel_field"],
        "id_field": cfg_data["id_field"],
        "skip_frames": cfg_data["skip_frames"],
    }
    data = SLPDataset(**data_path_kwarg, **data_params, load_only_skel=True)
    dataloader = build_slp_dataloader(
        dataset=data,
        shuffle=False,
        fixed_seq_len=cfg_model["T"],
        pad_value=pad_value, batch_size=batch_size,
    )

    # -- compute predictions
    predictions = {}
    skelmotionvae.eval()
    for batch in tqdm(dataloader):
        # x = batch["skels"].to("cpu")  # (B, T, Npts, 3)
        x = batch["skels"].to(device)  # (B, T, Npts, 3)
        ids = batch["id"]
        B, T = x.shape[:2]
        x_pad_mask = ~(x.view(B, T, -1) != pad_value).any(dim=-1)  # (B, T) | boolean (True=>pad)
        x_recon, _, _, _ = skelmotionvae(x, decode_mu=True, pad_mask=x_pad_mask)
        x_recon = x_recon.cpu()  # to cpu (more flexible for later)
        for i in range(x_recon.size(0)):
            predictions[ids[i]] = x_recon[i].clone()  # (T, Npts, 3)

    torch.save(predictions, f"{predictions_folder}/{data_name}_predictions.pt")

    return predictions
