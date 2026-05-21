import argparse
import os
import yaml
from pathlib import Path

from .trainers.train_autoencoder import train_skelmotionvae, compute_skelmotionvae_predictions

import random
import torch

from utils.visualization.make_video import make_video_simple
from utils.visualization.skels_def import MP178_CONNECTIONS, MP178_LH_IDS, MP178_RH_IDS, MP178_FACE_IDS


def main():
    ap = argparse.ArgumentParser(f"LatSkelClipDiff-Modules")

    # Always required
    ap.add_argument("--mode", choices=["train", "predict", "videos", "eval"], required=True)
    ap.add_argument("--cfg_path", type=str, required=True)

    # Required for 'predict' mode
    ap.add_argument("--ckpt_path", type=str, required=False)  # only required for 'predict' mode
    ap.add_argument("--data_name", type=str, required=False)  # only required for 'predict' mode
    # Optional for 'predict' mode
    ap.add_argument("--batch_size", type=int, default=64)  # optionally required for 'predict' mode
    ap.add_argument("--output_folder", type=str, required=False)  # optionally required for 'predict' mode

    # Required for 'eval' mode
    ap.add_argument("--eval_name", type=str, default="default")  # optionally required for 'eval' mode
    ap.add_argument("--eval_cfg_path", type=str, default=None)  # optionally required for 'eval' mode

    # Additionally required for 'videos' mode (temp)
    ap.add_argument("--pred_data_filepath", type=str)


    args = ap.parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    output_folder = args.output_folder if args.output_folder else cfg["training"]["output_dir"]

    # ===== SkelMotionVAE [training OR predictions OR videos OR evaluation]
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


if __name__ == "__main__":
    main()
