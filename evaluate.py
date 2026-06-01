"""
Main script for Sign Language Production performance evaluation.
"""

import argparse

from .utils.str2bool import str2bool
from .evaluation import evaluate


def main():
    ap = argparse.ArgumentParser("SLP Evaluation")

    # ------------------ ARGUMENTS
    # ****** Type of evaluation pipeline to use (among those available...)
    ap.add_argument("--evaluator", type=str, default="EvaluatorP14TvSLRTP25")

    # ****** Name to give to the evaluation + parent folder
    ap.add_argument("--eval_name", type=str, default="base")
    ap.add_argument("--output_folder", type=str, default="./slp_eval_results")

    # ****** Where to retrieve input data (ground truth data and pre-generated poses)
    ap.add_argument("--ground_truth_data_filepath", type=str)
    ap.add_argument("--predicted_data_filepath", type=str, default=None)
    ap.add_argument("--convert_to_compatible_pred", type=str2bool, default=False)

    ap.add_argument("--predicted_data_folder", type=str)
    ap.add_argument("--predicted_poses_field", type=str, default="poses_3d_gen")
    ap.add_argument("--predicted_id_field", type=str, default="id")

    # ****** Back-translation (Sign Language-to-Text) model
    ap.add_argument(
        "--backtranslation_model_dir",
        type=str, default="utils/slt_models/slrtp25_bt_phoenix14t"
    )  # WARNING: the default Phix14T BT model need 12 FPS inputs!!!!

    # ****** Pre-processing of sign poses sequences
    ap.add_argument("--ground_truth_skip_frames", type=int, default=2)  # WARNING: 2 if 25 FPS to match 12 FPS!!!!
    ap.add_argument("--predicted_skip_frames", type=int, default=2)  # WARNING: 2 if 25 FPS to match 12 FPS!!!!
    ap.add_argument("--cut_to_shorter_sequence", type=str2bool, default=True)
    ap.add_argument("--apply_savgol_smoothing", type=str2bool, default=False)
    ap.add_argument("--savgol_window", type=int, default=11)
    ap.add_argument("--savgol_poly", type=int, default=3)
    ap.add_argument("--store_per_sample_results", type=str2bool, default=False)
    ap.add_argument("--ensure_consistent_gt_orientation", type=str2bool, default=False)

    # ------------------ RETRIEVE ARGUMENTS & RUN EVALUATION
    args = ap.parse_args()

    print(f"Attempting to perform evaluation with evaluator: '{args.evaluator}'...")

    evaluate(
        evaluator_name=args.evaluator,
        name=args.eval_name,
        output_folder=args.output_folder,
        gt_data_filepath=args.ground_truth_data_filepath,
        pred_data_filepath=args.predicted_data_filepath,
        convert_to_compatible_pred=args.convert_to_compatible_pred,
        pred_data_folder=args.predicted_data_folder,
        pred_poses_field=args.predicted_poses_field,
        pred_id_field=args.predicted_id_field,
        bt_model_dir=args.backtranslation_model_dir,
        # -- prepro steps
        gt_skip_frames=args.ground_truth_skip_frames,
        pred_skip_frames=args.predicted_skip_frames,
        cut_to_shorter=args.cut_to_shorter_sequence,
        ensure_consistent_gt_orientation=args.ensure_consistent_gt_orientation,
        savgol_smooth=args.apply_savgol_smoothing,
        savgol_window=args.savgol_window,
        savgol_poly=args.savgol_poly,
        # -- extra folders
        store_sample_vals=args.store_per_sample_results,
    )

    print(f"Evaluation ended!\nFind results at: {args.output_folder}")


if __name__ == "__main__":
    main()
