import argparse
import os
import yaml
from pathlib import Path

from .utils.str2bool import str2bool
from .trainers.train_autoencoder import train_skelmotionvae, compute_skelmotionvae_predictions

import random
import torch
import sys

# ************** temp imports for video generation **************
sys.path.append("../..")
from utils.visualization.make_video import make_video_simple
from utils.visualization.skels_def import MP178_CONNECTIONS, MP178_LH_IDS, MP178_RH_IDS, MP178_FACE_IDS
# ***************************************************************


def main():
    ap = argparse.ArgumentParser(f"LatSkelClipDiff-Modules")

    # Always required
    ap.add_argument(
        "--model",
        choices=["skelmotionvae", "latentslclip", "text2latentsldiff"],
        required=True
    )
    ap.add_argument("--mode", choices=["train", "predict", "videos", "eval"], required=True)
    ap.add_argument("--cfg_path", type=str, required=True)

    # Required for 'predict' mode
    ap.add_argument("--ckpt_path", type=str, required=False)  # only required for 'predict' mode
    ap.add_argument("--data_name", type=str, required=False)  # only required for 'predict' mode
    # Optional for 'predict' mode
    ap.add_argument("--batch_size", type=int, default=64)  # optionally required for 'predict' mode
    ap.add_argument("--output_folder", type=str, required=False)  # optionally required for 'predict' mode
    ap.add_argument("--guidance_scale", type=float, default=2.)  # ONLY for 'text2latentsldiff' model (LDM) | optional
    ap.add_argument("--use_bert_embeddings", type=str2bool,
                    default=False)  # ONLY for 'text2latentsldiff' model (LDM) if used for training

    # Required for 'eval' mode
    ap.add_argument("--clip_predictions_path", type=str, required=False)  # only required for 'eval' mode
    ap.add_argument("--eval_name", type=str, default="default")  # optionally required for 'eval' mode
    ap.add_argument("--eval_cfg_path", type=str, default=None)  # optionally required for 'eval' mode

    # Additionally required for 'videos' mode (temp)
    ap.add_argument("--pred_data_filepath", type=str)


    args = ap.parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    output_folder = args.output_folder if args.output_folder else cfg["training"]["output_dir"]

    # ===== SkelMotionVAE [training OR predictions OR videos OR evaluation]
    if args.model == "skelmotionvae":

        # >> TRAINING ------------------------
        if args.mode == "train":
            train_skelmotionvae(cfg)

        # >> PREDICTIONS ---------------------
        elif args.mode == "predict":
            os.makedirs(output_folder, exist_ok=True)
            compute_skelmotionvae_predictions(
                output_folder=output_folder,
                cfg_model=cfg["model"],
                ckpt_path=args.ckpt_path,
                cfg_data=cfg["data"],
                data_name=args.data_name,
                batch_size=args.batch_size,
            )

        # >> VIDEOS -------------------------
        elif args.mode == "videos":  # temp implementation of this mode for video generation
            # load gt
            data_name = args.data_name
            gt_data = torch.load(cfg["data"][data_name]["path"])
            # load predictions
            # pred_data = torch.load(cfg["training"]["output_dir"] + "/predictions_best/test_predictions.pt")
            pred_data_fpath = args.pred_data_filepath
            pred_data = torch.load(pred_data_fpath)
            # get ids
            N = 10
            random.seed(42)
            videos_ids = random.sample(list(gt_data.keys()), N)
            # videos folder
            # videos_folder = cfg["training"]["output_dir"] + "/videos/predictions_best/test_predictions"
            videos_folder = cfg["training"][
                                "output_dir"] + f"/videos/{Path(pred_data_fpath).parent.stem}/{data_name}_predictions"
            os.makedirs(videos_folder, exist_ok=True)
            for vid_id in videos_ids:
                skel_pred = pred_data[vid_id]
                skel_gt = gt_data[vid_id]["poses_3d"][:skel_pred.shape[0]]
                # invert y
                skel_pred[:, :, 1] *= -1  # invert y
                skel_gt[:, :, 1] *= -1  # invert y
                # make vid
                make_video_simple(
                    out_file=videos_folder + f"/{vid_id}.mp4",
                    seq1=skel_gt.numpy(), seq2=skel_pred.numpy(),
                    bones=MP178_CONNECTIONS,
                    no_link_ids=MP178_FACE_IDS,
                    face_ids=MP178_FACE_IDS, rh_ids=MP178_RH_IDS, lh_ids=MP178_LH_IDS,
                    lims=None,
                    text_str=gt_data[vid_id]["text"],
                    fps=25, dpi=140,
                )

        # >> EVALUATION -------------------------
        elif args.mode == "eval":
            raise NotImplementedError

        else:
            raise ValueError(f"Unknown mode '{args.mode}' for model={args.model}")

    # ===== LatentSignLanguageCLIP [training OR predictions OR evaluation]
    elif args.model == "latentslclip":

        # >> TRAINING ------------------------
        if args.mode == "train":
            train_latentslclip(cfg)

        # >> PREDICTIONS ---------------------
        elif args.mode == "predict":
            os.makedirs(output_folder, exist_ok=True)
            compute_latentslclip_predictions(
                output_folder=output_folder,
                cfg_model=cfg["model"],
                cfg_pretrained_vae=cfg["model"]["vae"],
                ckpt_path=args.ckpt_path,
                vae_ckpt_path=cfg["model"]["vae"]["ckpt"],
                cfg_data=cfg["data"],
                data_name=args.data_name,
                batch_size=args.batch_size,
            )

        # >> EVALUATION ---------------------
        elif args.mode == "eval":
            os.makedirs(output_folder, exist_ok=True)
            default_predictions_path = f"{output_folder}/predictions_best/{args.data_name}_predictions.pt"
            predictions_path = args.clip_predictions_path if args.clip_predictions_path else default_predictions_path
            cfg_eval = None
            if args.eval_cfg_path:
                with open(args.eval_cfg_path, "r") as f:
                    cfg_eval = yaml.safe_load(f)
            evaluate_latentslclip(
                output_folder=output_folder,
                eval_name=args.eval_name,
                cfg_eval=cfg_eval,
                predictions_path=predictions_path,
                cfg_model=cfg["model"],
                cfg_data=cfg["data"],
                data_name=args.data_name,
            )

        else:
            raise ValueError(f"Unknown mode '{args.mode}' for model={args.model}")

    # ===== Text2LatentSignLanguageDiffusion [training OR predictions OR evaluation]
    elif args.model == "text2latentsldiff":

        # >> TRAINING ------------------------
        if args.mode == "train":
            train_text2latentsldiffusion(cfg)

        # >> PREDICTIONS ---------------------
        elif args.mode == "predict":
            os.makedirs(output_folder, exist_ok=True)
            output_shape = (cfg["model"]["T"], cfg["model"]["dim_lat"])
            beta_schedule_kwargs = {
                "beta_schedule": cfg["training"].get("beta_schedule", "linear"),
                "beta_start": cfg["training"].get("beta_start", 1.e-4),
                "beta_end": cfg["training"].get("beta_end", 2e-2),
            }
            latent_stats_path = output_folder + "/latent_stats.pt"
            compute_text2latentsldiffusion_predictions(
                output_folder=output_folder,
                cfg_model=cfg["model"],
                output_shape=output_shape,
                w_cfg=args.guidance_scale,
                use_txt_bert=args.use_bert_embeddings,
                num_timesteps=cfg["training"]["diffusion_steps"],
                **beta_schedule_kwargs,
                ckpt_path=args.ckpt_path,
                latent_stats_path=latent_stats_path,
                cfg_data=cfg["data"],
                data_name=args.data_name,
                batch_size=args.batch_size,
            )

        # >> EVALUATION ---------------------
        elif args.mode == "eval":
            raise NotImplementedError

        else:
            raise ValueError(f"Unknown mode '{args.mode}' for model={args.model}")

    else:
        raise ValueError(f"Unknown model '{args.model}'")


if __name__ == "__main__":
    main()
