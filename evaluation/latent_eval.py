import os
import torch
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from typing import Tuple, List, Union

from ..data.dataset import SLPDataset, build_slp_dataloader
from ..model.autoencoders import load_vae
from ..utils.logs_checkpoints import load_ckpt


class LatentSpaceAnalyzer:
    """
    Utility class for analyzing latent representations produced by a trained VAE.
    The analyzer computes statistical and temporal metrics over the latent space
    of a dataset encoded with a pretrained model.

    See the project README for a detailed description of the available metrics.
    """
    _default_metrics = (
        "latent_smoothness",
        "covariance_matrix",
        "latent_energy"
    )

    def __init__(self):
        pass

    def run(
            self,
            cfg_model: dict,  # VAE config (dictionary)
            ckpt_path: str,  # path to trained VAE checkpoint
            cfg_data: dict,  # data config (dictionary)
            output_folder: str,
            name: str,
            metrics: tuple = None,
            plot_correl: bool = False,
            batch_size: int = 256,
            device: str = "cuda",
    ):
        """
        Run latent-space analysis on a pretrained VAE and dataset.

        This method:
            1. Loads a pretrained VAE checkpoint.
            2. Builds the dataset and dataloader.
            3. Encodes samples into the latent space.
            4. Computes the requested latent-space metrics.
            5. Saves the aggregated results as a JSON file.

        Args:
            cfg_model (dict):
                Model configuration dictionary used to instantiate the VAE.
                See the ROOT/configs examples for supported fields and parameters.

            ckpt_path (str):
                Path to the pretrained VAE checkpoint.

            cfg_data (dict):
                Dataset configuration dictionary.
                See ROOT/configs examples for supported fields and dataset properties.

            output_folder (str):
                Directory where analysis outputs and result files are saved.

            name (str):
                Experiment or model name used when naming output files.

            metrics (tuple, optional):
                Tuple of metric names to compute. If ``None``, default metrics defined
                in ``_default_metrics`` are used.

            plot_correl (bool, optional):
                Whether to save latent-dimension correlation matrix heatmaps for each
                processed batch. Defaults to ``False``.

            batch_size (int, optional):
                Batch size used during latent extraction. Defaults to ``256``.

            device (str, optional):
                Device used for inference (e.g. ``"cuda"`` or ``"cpu"``).
                Defaults to ``"cuda"``.

        Returns:
            dict:
                Dictionary containing the aggregated latent-space metrics computed
                over the dataset.
        """
        os.makedirs(output_folder, exist_ok=True)

        # === 1.a) Load the VAE and freeze its weights
        print("1.a - Loading Sign Pose VAE...")

        ckpt = load_ckpt(ckpt_path, device=device)
        skelmotionvae = load_vae(name=cfg_model.get("name", "SkelMotionMultiRegionVAE"), cfg=cfg_model)
        skelmotionvae.load_state_dict(ckpt["model_state_dict"], strict=False)
        skelmotionvae = skelmotionvae.to(device=device)

        print("Done.\n")

        # === 1.b) Load dataset & build dataloader
        print("1.b - Loading dataset and building dataloader...")

        data_path = cfg_data["path"]
        path_kwarg = {f"{cfg_data['type']}_path": data_path}
        data_params = {
            "skel_field": cfg_data["skel_field"],
            "id_field": cfg_data["id_field"],
            "skip_frames":cfg_data["skip_frames"],
        }
        data = SLPDataset(**path_kwarg, **data_params, load_only_skel=True)

        pad_value = 0.0
        dataloader_params = {
            "pad_value": pad_value,
            "batch_size": batch_size,
            "fixed_seq_len": cfg_model["T"],
        }
        dataloader = build_slp_dataloader(dataset=data, shuffle=False, **dataloader_params)

        print("Done.\n")

        # === 2) Compute metrics over the latent dataset
        if metrics is None:
            metrics = self._default_metrics

        print(f"2 - Computing the following metrics on latent space: {metrics}")

        n_tot = 0
        n_batches = 0

        # vel / acc
        v_tot = 0.
        a_tot = 0.

        # "energy" (mean squared norm)
        e_tot = 0.

        # covariance metrics
        effective_dim_tot = 0.
        anisotropy_ratio_tot = 0.
        mean_abs_offdiag_corr_tot = 0.

        for batch in tqdm(dataloader, desc="Metrics computation over the dataset..."):
            poses = batch["skels"].to(device)  # (B, T, N, 3)
            B, T = poses.shape[:2]
            poses_pad_mask = ~(poses.view(B, T, -1) != pad_value).any(dim=-1)  # (B, T) | boolean (True=>pad)
            mu, _, _ = skelmotionvae.encode(X=poses, pad_mask=poses_pad_mask)
            n_tot += len(batch)
            n_batches += 1
            for m in metrics:
                if m == "latent_smoothness":
                    # sum over batch samples
                    v_batch, a_batch = self.temporal_smoothness(mu, ~poses_pad_mask)
                    v_tot += v_batch
                    a_tot += a_batch
                elif m == "latent_energy":
                    # sum over batch samples
                    e_tot += self.latent_energy(mu, ~poses_pad_mask)
                elif m == "covariance_matrix":
                    # already per-batch values
                    covariance_metrics = self.empirical_covariance_matrix(mu, ~poses_pad_mask)
                    effective_dim_tot += covariance_metrics["effective_dim"]
                    anisotropy_ratio_tot += covariance_metrics["anisotropy_ratio"]
                    mean_abs_offdiag_corr_tot += covariance_metrics["mean_abs_offdiag_corr"]
                    if plot_correl:
                        correl_folder = f"{output_folder}/correlation_matrices"
                        correl_fname = f"dims_correl_batch{n_batches}.png"
                        self._plot_square_matrix_heatmap(
                            mat=covariance_metrics["correlation"],
                            output_folder=correl_folder,
                            filename=correl_fname,
                            title=f"Correlation matrix between latent dimensions over batch n°{n_batches}",
                            x_label=r"$Latent$ $dimensions$",
                            y_label=r"$Latent$ $dimensions$",
                        )
                        print(f"Saved correlation matrix to: {correl_folder}/{correl_fname} ")
                else:
                    raise NotImplementedError(f"Unknown metric '{m}'. Valid metrics are: {self._default_metrics}")

        # average over dataset or batches (depending on the metrics)
        results = {}
        for m in metrics:
            if m == "latent_smoothness":
                results[m] = {"vel": v_tot / n_tot, "acc": a_tot / n_tot}
            elif m == "latent_energy":
                results[m] = {"energy": e_tot / n_tot}
            elif m == "covariance_matrix":
                results[m] = {
                    "eff_dim": effective_dim_tot / n_batches,
                    "aniso": anisotropy_ratio_tot / n_batches,
                    "mean_abs_corr": mean_abs_offdiag_corr_tot / n_batches,
                }
            else:
                raise NotImplementedError(f"Unknown metric '{m}'.")

        # === 3) Save results to a .json file
        results_fpath = f"{output_folder}/{name}_LatentAnalysis.json"
        with open(results_fpath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print("Done.")
        print(f"Saved results to {results_fpath}")

        return results

    @staticmethod
    def _plot_square_matrix_heatmap(
            mat: torch.Tensor,
            output_folder: str,
            filename: str,
            title: str = "Heatmap",
            x_label: str = None,
            y_label: str = None,
            figsize: tuple = (8, 6),
            cmap: str = "coolwarm",
    ):
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, filename)

        d = mat.shape[0]
        mat_np = mat.cpu().numpy()

        plt.figure(figsize=figsize)

        sns.heatmap(
            mat_np,
            cmap=cmap,
            center=0,
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'shrink': 0.8, 'label': 'Value'},
            annot=False  # set True if you want numbers in cells
        )

        plt.title(title, fontsize=16, fontweight='bold')
        if x_label:
            plt.xlabel(x_label, fontsize=18)
        if y_label:
            plt.ylabel(y_label, fontsize=18)

        if x_label is not None:
            plt.xticks(
                ticks=np.arange(1, d, 2) + 0.5,
                labels=[str(i + 1) for i in range(1, d, 2)],
                fontsize=10,
                rotation=90,
                ha="right"
            )

        if y_label is not None:
            plt.yticks(
                ticks=np.arange(1, d, 2) + 0.5,
                labels=[str(i + 1) for i in range(1, d, 2)],
                fontsize=10,
                rotation=0
            )


        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def temporal_smoothness(
            batch_latent_poses: torch.Tensor,
            pad_mask: torch.Tensor,  # pad = False | valid frame = True
    ) -> Tuple[float, float]:
        """
        Computes:

                v = sum_i 1/(T_i - 1) sum_t || z{t+1} - z{t} ||_2  (1st order temporal smoothness)

                a = sum_i 1/(T_i - 2) sum_t || z{t+2} - 2 z{t+1} + z{t} ||_2  (2nd order temporal smoothness)

        """
        d = batch_latent_poses.shape[-1]  # latent dimension

        vel = batch_latent_poses[:, 1:] - batch_latent_poses[:, :-1]  # (B, T-1, d)
        acc = batch_latent_poses[:, 2:] - 2 * batch_latent_poses[:, 1:-1] + batch_latent_poses[:, :-2]  # (B, T-2, d)

        vel_norm = torch.norm(vel, p=2, dim=-1) / d  # (B, T-1)
        valid_vel_mask = pad_mask[:, :-1] & pad_mask[:, 1:]
        v = (vel_norm * valid_vel_mask).sum(dim=1) / (valid_vel_mask.sum(dim=1) + 1e-8)  # (B,)

        acc_norm = torch.norm(acc, p=2, dim=-1) / d  # (B, T-2)
        valid_acc_mask = pad_mask[:, :-2] & pad_mask[:, 1:-1] & pad_mask[:, 2:]
        a = (acc_norm * valid_acc_mask).sum(dim=1) / (valid_acc_mask.sum(dim=1) + 1e-8)  # (B,)

        return v.sum().item(), a.sum().item()

    @staticmethod
    def latent_energy(
            batch_latent_poses: torch.Tensor,
            pad_mask: torch.Tensor,  # pad = False | valid frame = True
    ) -> float:
        """
        Computes:

                e = sum_i 1/T_i sum_t ( || z_t ||_2 )^2
        """
        d = batch_latent_poses.shape[-1]

        energy = torch.norm(batch_latent_poses, p=2, dim=-1) ** 2 / (d ** 2)  # (B, T)

        valid_energy = energy * pad_mask  # (B, T)
        e = valid_energy.sum(dim=1) / (pad_mask.sum(dim=1) + 1e-8)  # (B,)

        return e.sum().item()

    @staticmethod
    def empirical_covariance_matrix(
            batch_latent_poses: torch.Tensor,
            pad_mask: torch.Tensor,  # pad = False | valid frame = True
    ) -> dict:
        """
        Computes empirical covariance, correlation matrix, eigenvalues, and basic stats
        over all valid latent frames in the batch.

        Returns a dictionary:
            'covariance': dxd covariance matrix
            'correlation': dxd correlation matrix
            'eigenvalues': d
            'effective_dim': scalar
            'anisotropy_ratio': scalar (max/min eigenvalue)
            'mean_abs_offdiag_corr': scalar (mean absolute off-diagonal correlation)
        """
        B, T, d = batch_latent_poses.shape

        # Mask invalid frames
        valid_latents = batch_latent_poses[pad_mask]  # (N_valid, d)

        # -- empirical mean
        mu = valid_latents.mean(dim=0, keepdim=True)  # (1, d)

        # -- covariance
        centered = valid_latents - mu  # (N_valid, d)
        N = centered.shape[0]
        cov = (centered.T @ centered) / (N - 1)  # (d, d)

        # -- correlation matrix
        std = torch.sqrt(torch.diag(cov) + 1e-8)  # avoid div by 0
        corr = cov / (std[:, None] * std[None, :])

        # -- eigenvalues
        eigvals = torch.linalg.eigvalsh(cov)  # (d,) ascending
        eigvals_sorted, _ = torch.sort(eigvals, descending=True)

        # -- effective dimensionality (sum_k lambda_k)^2 / sum_k lambda_k^2
        # => cf. DOI:10.1080/00273171.2020.1743631
        eff_dim = (eigvals.sum() ** 2) / (eigvals.pow(2).sum() + 1e-8)

        # -- anisotropy ratio lambda_max / lambda_min => how much in principal direction vs lower direction
        anisotropy = eigvals.max() / (eigvals.min() + 1e-8)

        # -- mean absolute off-diagonal correlation
        off_diag_mask = ~torch.eye(d, dtype=torch.bool, device=corr.device)
        mean_abs_offdiag = corr[off_diag_mask].abs().mean().item()

        return {
            'covariance': cov,
            'correlation': corr,
            'eigenvalues': eigvals_sorted,
            'effective_dim': eff_dim.item(),
            'anisotropy_ratio': anisotropy.item(),
            'mean_abs_offdiag_corr': mean_abs_offdiag,
        }


def evaluate_latent_space(
        output_folders: Union[str, List[str]],
        names: Union[str, List[str]],
        cfg_model_filepaths: Union[str, List[str]],
        ckpt_filepaths: Union[str, List[str]],
        cfg_data_filepaths: Union[str, List[str]],
        metrics: tuple = None,
        batch_size: int = 256,
        plot_correl: bool = False,
):
    # -- load model config(s)
    if isinstance(cfg_model_filepaths, str):
        cfg_model_filepaths = [cfg_model_filepaths]

    print("Loading model configs...")

    model_configs = []
    for cfg_model_fpath in cfg_model_filepaths:
        print(f"   - {cfg_model_fpath}")
        with open(cfg_model_fpath, "r") as f:
            model_configs.append(yaml.safe_load(f))

    # -- load data config(s)
    if isinstance(cfg_data_filepaths, str):
        cfg_data_filepaths = [cfg_data_filepaths]

    print("Loading dataset configs...")

    data_configs = []
    for cfg_data_fpath in cfg_data_filepaths:
        print(f"   - {cfg_data_fpath}")
        with open(cfg_data_fpath, "r") as f:
            data_configs.append(yaml.safe_load(f))

    # -- instantiate latent space analyzer
    analyzer = LatentSpaceAnalyzer()

    # -- analyze the latent data of the data config(s)
    if isinstance(output_folders, str):
        output_folders = [output_folders]
    if isinstance(names, str):
        names = [names]

    print("\nRunning latent space analysis for each config...")

    for cfg_model, ckpt_path, cfg_data, out_folder, name in zip(
            model_configs,
            ckpt_filepaths,
            data_configs,
            output_folders,
            names
    ):
        print(f"   ==> running for '{name}'...")
        analyzer.run(
            cfg_model=cfg_model,
            ckpt_path=ckpt_path,
            cfg_data=cfg_data,
            output_folder=out_folder,
            name=name,
            metrics=metrics,
            plot_correl=plot_correl,
            batch_size=batch_size,
        )
