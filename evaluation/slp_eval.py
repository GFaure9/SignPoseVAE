import sys
import glob
import torch
from pathlib import Path
from tqdm import tqdm

# ****************** temp imports for evaluation ******************
# (may re-write evaluator later)
# sys.path.append("../../..")
from utils.metrics.evaluation import EvaluatorP14TvSLRTP25


# *****************************************************************


def evaluate(
        output_folder: str,
        name: str,

        evaluator_name: str,

        # === ground-truth data
        gt_data_filepath: str,
        # gt_poses_field: str = "poses_3d",
        # gt_text_field: str = "text",

        # === predicted data
        pred_data_filepath: str = None,
        convert_to_compatible_pred: bool = False,

        pred_data_folder: str = None,
        pred_poses_field: str = "poses_3d_gen",
        pred_id_field: str = "id",

        # === back-translation model directory
        bt_model_dir: str = None,

        # === whether to save per sample data
        store_sample_vals: bool = False,

        # === pre-processing steps
        gt_skip_frames: int = 1,
        pred_skip_frames: int = 1,
        cut_to_shorter: bool = True,
        savgol_smooth: bool = False,
        savgol_window: int = 11,
        savgol_poly: int = 3,
        ensure_consistent_gt_orientation: bool = False,
):
    """
    Function to evaluate the final generated sign poses versus ground truth data in terms of
    Sign Language Production performance.

    Implements the following evaluation pipelines:

        >> 'EvaluatorP14TvSLRTP25':
            * Expected skeletal data format: CVPR 2025 SLRTP challenge format
                                            (128 face + 50 body/arms/hands = 178 key-points skeleton)
            * Expected ground truth data: Phoenix14T
            * Expected back-translation model:
                https://github.com/walsharry/SLRTP-Sign-Production-Evaluation/tree/main/back_translation
                (with pre-trained model at https://drive.google.com/file/d/1fjKHigsEWHwsMHnwwWdFYZ8dECXslTKi/view)
            * Computed metrics: (included CVPR 2025 SLRTP challenge metrics and other)
                - geometric metrics: DTWp, DTW-MJE, DTW-MBAE-BODY, DTW-MBAE-LH, DTW-MBAE-RH, DTW-PCK, TotalDistance
                - BT metrics: BLEU1-4, ROUGE-L, chrF, WER

        >> More evaluations might be coming in the future...

    """

    if evaluator_name == "EvaluatorP14TvSLRTP25":
        # Prepare predicted data format for EvaluatorP14TvSLRTP25 compatibility
        if pred_data_filepath is None:

            if pred_data_folder is not None:
                # Assumes that the given folder contains .pt files
                # which names are examples ids and which contain the key `pred_pose_field`
                # associated to a torch.Tensor value.

                # -- create a .pt file in the format:
                #       {example_id1: torch.Tensor, ..., example_idN: torch.Tensor}
                filepaths = glob.glob(f"{pred_data_folder}/*.pt")
                predicted_data_dic = {}

                for fpath in tqdm(filepaths, desc="Creating dictionary of predicted data..."):
                    prediction = torch.load(fpath, weights_only=True)
                    example_id = prediction.get(pred_id_field, Path(fpath).stem)
                    predicted_data_dic[example_id] = prediction[pred_poses_field]

                pred_data_filepath = f"{Path(pred_data_folder).parent}/{Path(pred_data_folder).stem}.pt"
                torch.save(predicted_data_dic, pred_data_filepath)
                print(f"Saved predictions from {pred_data_folder} into a single file: {pred_data_filepath}")

            else:
                raise ValueError(
                    "You must either pass `pred_data_filepath` or `pred_data_folder` as argument"
                )

        else:
            if convert_to_compatible_pred:
                pred_data = torch.load(pred_data_filepath, weights_only=True)
                predicted_data_dic = {}

                for k, v in tqdm(pred_data.items(), total=len(pred_data),
                                 desc="Creating dictionary of predicted data..."):
                    example_id = v.get(pred_id_field, k)
                    predicted_data_dic[example_id] = v[pred_poses_field]

                pred_data_filepath = f"{Path(pred_data_filepath).parent}/{Path(pred_data_filepath).stem}_compatible.pt"
                torch.save(predicted_data_dic, pred_data_filepath)
                print(f"Saved predictions from {pred_data_folder} into a single file: {pred_data_filepath}")

                # Define pre-processing steps
        preprocessing_bt = {
            "skip_frames": {
                "groundtruth": gt_skip_frames,
                "preds": pred_skip_frames,
            },
            "cut_to_shorter": cut_to_shorter,
            "ensure_consistent_gt_orientation": ensure_consistent_gt_orientation,
        }

        # ---- FOR GEOMETRIC METRICS NO FRAME IS SKIPPED
        preprocessing_geo = {
            "skip_frames": {
                "groundtruth": gt_skip_frames,  # 1,
                "preds": pred_skip_frames,  # 1,
            },
            "cut_to_shorter": cut_to_shorter,
            "ensure_consistent_gt_orientation": ensure_consistent_gt_orientation,
        }

        # (optional) apply  Savitzky-Golay filter for temporal smoothing - only on predictions
        if savgol_smooth:
            temp_smooth = {
                "method": "savgol",
                "kwargs": {"window": savgol_window, "poly": savgol_poly}
            }
            preprocessing_bt["temp_smooth"] = temp_smooth
            preprocessing_geo["temp_smooth"] = temp_smooth

        # Compute evaluations & store results
        assert bt_model_dir is not None, "Must provide a back-translation (BT) model directory to perform BT eval"
        evaluator = EvaluatorP14TvSLRTP25(bt_model_dir=bt_model_dir)

        results_geo = evaluator.run_(
            preds_fpath=pred_data_filepath,
            groundtruth_fpath=gt_data_filepath,
            output_folder=output_folder,
            name=name + f"skipframes_{pred_skip_frames}",
            metrics_type="geo",
            store_sample_vals=store_sample_vals,
            **preprocessing_geo
        )

        results_bt = evaluator.run_(
            preds_fpath=pred_data_filepath,
            groundtruth_fpath=gt_data_filepath,
            output_folder=output_folder,
            name=name + f"skipframes_{gt_skip_frames}",
            metrics_type="bt",
            store_sample_vals=store_sample_vals,
            **preprocessing_bt
        )

        print(name + f" - {evaluator_name} evaluation results GEO:\n", results_geo)
        print(name + f" - {evaluator_name} evaluation results BT:\n", results_bt)

    else:
        raise NotImplementedError(f"Unknown evaluator name '{evaluator_name}'")
