import torch
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from time import time
from typing import Dict, Union, Tuple, List
from utils.metrics.evaluation.geometric_metrics import (
    dtwp, dtw_align, mpje, mpbae, mpboe, pck, total_distance
)
from utils.metrics.evaluation.bones_for_angles_slrtp25 import (
    BONES_FOR_ANGLES_SLRTP25,
    RH_BONES_FOR_ANGLES_SLRTP25,
    LH_BONES_FOR_ANGLES_SLRTP25,
    HANDS_BONES_FOR_ANGLES_SLRTP25,
    BODY_BONES_FOR_ANGLES_SLRTP25
)
from utils.slt_models.slrtp25_bt_phoenix14t.back_translate import back_translate, make_back_translation_model
from utils.metrics.slrtp_challenge_2025_evaluation.metrics import (
    wer,
    bleu,
    chrf,
    rouge,
)


def timer(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2 - t1) // 60}min {(t2 - t1) % 60:.4f}s")
        return result

    return wrap_func


def savgol(X: torch.Tensor, window=11, poly=3):
    # if X.shape[0] < window:
    #     return X
    # else:
    #     X_np = X.detach().cpu().numpy()  # shape=(T, N, 3)
    #     Y_np = savgol_filter(
    #         X_np,
    #         window_length=window,
    #         polyorder=poly,
    #         axis=0,  # smooth over time
    #         mode='interp'
    #     )
    #     return torch.from_numpy(Y_np).to(X.device)

    w = min(window, X.shape[0])
    X_np = X.detach().cpu().numpy()  # shape=(T, N, 3)
    Y_np = savgol_filter(
        X_np,
        window_length=w,
        polyorder=poly,
        axis=0,  # smooth over time
        mode='interp'
    )
    return torch.from_numpy(Y_np).to(X.device)


class EvaluatorP14TvSLRTP25:
    GEO_METRICS = [
        "DTWp",
        "DTW-MJE",
        # **************
        # Warning: commented because GT vs GT with these metrics
        #          do not give 0.0 due to numerical precision issues
        #          with torch.acos()
        # "DTW-MBAE-BODY",
        # "DTW-MBAE-RH",
        # "DTW-MBAE-LH",
        # **************
        "DTW-MBOE-BODY",
        "DTW-MBOE-RH",
        "DTW-MBOE-LH",
        "DTW-PCK",
        "TotalDistance",
    ]
    BT_METRICS = [
        "BLEU",
        "ROUGE-L",
        "chrF",
        "WER",
        "Likelihood",
    ]

    def __init__(self, bt_model_dir: str = None):
        self.preds = None
        self.groundtruth = None
        self.eval = None
        self.bt_model_dir = bt_model_dir

    @property
    def num_keypoints(self) -> int:
        return 178

    @property
    def rh_ids(self) -> list:
        return list(range(8, 29))

    @property
    def lh_ids(self) -> list:
        return list(range(29, 50))

    @property
    def face_ids(self) -> list:
        return list(range(50, 178))

    @timer
    def load_predictions(self, fpath: str) -> Dict[str, torch.Tensor]:
        """
        Load predictions made with an SLP model, stored in a .pt file as a dictionnary of
        the form:
                    {
                        EXAMPLE_NAME_1: torch.Tensor(T_1, 178, 3),
                        ...
                        EXAMPLE_NAME_k: torch.Tensor(T_k, 178, 3),
                        ...
                        EXAMPLE_NAME_Ntot: torch.Tensor(T_Ntot, 178, 3)
                    }
        """
        preds = torch.load(fpath, weights_only=True, map_location="cpu")

        # handle the case where dict keys are of the form {SUBSET}/{EXAMPLE_NAME_k}
        preds = {k.split("/")[-1]: v for k, v in preds.items()}

        if self.preds is None:
            self.preds = preds.copy()

        return preds

    @timer
    def load_groundtruth(self, fpath: str) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Load ground truth data, stored in a .pt file as a dictionnary of
        the form:
                    {
                        EXAMPLE_NAME_1: {'text': str, 'poses_3d': torch.Tensor(T_1, 178, 3)},
                        ...
                        EXAMPLE_NAME_k: {'text': str, 'poses_3d': torch.Tensor(T_k, 178, 3)},
                        ...
                        EXAMPLE_NAME_Ntot: {'text': str, 'poses_3d': torch.Tensor(T_Ntot, 178, 3)}
                    }
        Note that ground truth data keys (example names or ids) must match predictions ones.
        """
        groundtruth = torch.load(fpath, weights_only=True, map_location="cpu")

        # handle the case where dict keys are of the form {SUBSET}/{EXAMPLE_NAME_k}
        groundtruth = {k.split("/")[-1]: v for k, v in groundtruth.items()}

        assert {"text", "poses_3d"}.issubset(
            groundtruth[next(iter(groundtruth))].keys()), "Examples dicts must have 'text' and 'poses_3d' keys"

        if self.groundtruth is None:
            self.groundtruth = groundtruth.copy()

        return groundtruth

    @staticmethod
    def preprocessing(preds, groundtruth, **prepro_steps) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, Union[str, torch.Tensor]]]:
        """In-place prepro. Return preprocessed inputs."""

        for step, val in prepro_steps.items():

            if step == "skip_frames":
                print(f"skipping frames:\n{val}")
                if "preds" in val.keys():
                    for k, v in preds.items():
                        preds[k] = v[::val["preds"]]
                if "groundtruth" in val.keys():
                    for k, v in groundtruth.items():
                        groundtruth[k]["poses_3d"] = v["poses_3d"][::val["groundtruth"]]

            if step == "cut_to_shorter":
                if val:
                    print(f"cutting the longer sequence to match the shorter one")
                    for k in groundtruth.keys():
                        L = min(len(groundtruth[k]["poses_3d"]), len(preds[k]))
                        groundtruth[k]["poses_3d"] = groundtruth[k]["poses_3d"][:L]
                        preds[k] = preds[k][:L]

            if step == "ensure_consistent_gt_orientation":
                # making sure no groundtruth pose is upside down
                # ie making sure that RHip -> RShoulder always points to -Y (normal orientation in the dataset)
                print(
                    f"making sure no groundtruth pose sequence is upside down (flipping poses vertically if it's the case)")
                RHipID, RShoulderID = 6, 0
                num_flipped = 0
                for k in groundtruth.keys():
                    gt_poses = groundtruth[k]["poses_3d"]
                    orientation_vec = gt_poses[:, RShoulderID] - gt_poses[:, RHipID]  # (T, 3)
                    is_flipped = torch.mean(orientation_vec[:, 1]) > 0  # mean of y values over frames
                    if is_flipped:
                        groundtruth[k]["poses_3d"][:, :, 1] *= -1
                        num_flipped += 1
                print("---> Number of flipped gt sequence = ", num_flipped)

            if step == "temp_smooth":
                # {'temp_smooth': {'method': FILTER_NAME, "kwargs": KWARGS_DICT}}
                temp_smooth_method = val.get("method", None)
                temp_smooth_kwargs = val.get("kwargs", {})
                if temp_smooth_method is not None:
                    print(f"Temporally smoothing predictions using: '{val}' filter")
                    if len(temp_smooth_kwargs) > 0:
                        print(f"With arguments: {temp_smooth_kwargs}")
                    if temp_smooth_method == "savgol":
                        for k in groundtruth.keys():
                            preds[k] = savgol(preds[k], **temp_smooth_kwargs)
                    else:
                        raise NotImplementedError("Unknown method")
                else:
                    print("No method was given for temporal smoothing. Skipping it.")

        return preds, groundtruth

    @staticmethod
    def get_data_lists(preds, groundtruth) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
        preds_list = []
        groundtruth_list = []
        text_groundtruth = []
        ids = []
        for ex in groundtruth.keys():
            preds_list.append(preds[ex])
            groundtruth_list.append(groundtruth[ex]["poses_3d"])
            text_groundtruth.append(groundtruth[ex]["text"])
            ids += [ex]
        return preds_list, groundtruth_list, text_groundtruth, ids

    @timer
    def evaluate(self, preds, groundtruth, metrics, sample_vals_folder=None) -> dict:
        print("Writing (pred & gt) skels and (gt) text to lists...")
        preds_list, groundtruth_list, text_groundtruth, ex_ids = self.get_data_lists(preds, groundtruth)
        assert len(preds_list) == len(groundtruth_list)
        print(f"Number of sequences = {len(groundtruth_list)}")

        print("Computing evaluation for the following metrics:")
        for m in metrics:
            print(f"     - {m}")

        text_preds = None
        if set(metrics) & set(self.BT_METRICS):
            print(f"Existing BT metrics, computing backtranslation using sign-to-text model at: {self.bt_model_dir}")
            if self.bt_model_dir is not None:
                bt_model = make_back_translation_model(model_dir=self.bt_model_dir)
                text_preds = back_translate(model=bt_model, poses=preds_list)
            else:
                raise ValueError(
                    "No `bt_model_dir` was provided when instantiating evaluator. Required for BT metrics.")

        if sample_vals_folder is not None and text_preds is not None:
            assert len(text_preds) == len(ex_ids)
            per_sample_bt_text = [
                {"id": ex_ids[k], "groundtruth_text": text_groundtruth[k], "bt_text": text_preds[k]}
                for k in range(len(ex_ids))
            ]
            with open(f"{sample_vals_folder}/bt_text.json", "w", encoding="utf-8") as f:
                json.dump(per_sample_bt_text, f, ensure_ascii=False, indent=4)

        eval = {}
        for m in metrics:
            print(f">>>> Computing {m}")

            # =============== DTWp
            if m == "DTWp":
                eval[m], _ = dtwp(gt_poses=groundtruth_list, hypo_poses=preds_list, fps=25)

            # =============== Mean per joint error after joint distance-based DTW alignment
            if m == "DTW-MJE":
                aligned_grountruth_list, aligned_preds_list, _ = dtw_align(
                    gt_poses=groundtruth_list,
                    hypo_poses=preds_list,
                    criterion="joint_distance",
                )
                eval[m], all_mje = mpje(gt_poses=aligned_grountruth_list, hypo_poses=aligned_preds_list)
                print("Mean Per Joint Position Error: ", eval[m])
                if sample_vals_folder is not None:
                    assert len(all_mje) == len(ex_ids)
                    per_sample_mje = [
                        {"id": ex_ids[k], "mje": all_mje[k]}
                        for k in range(len(ex_ids))
                    ]
                    with open(f"{sample_vals_folder}/mje.json", "w") as f:
                        json.dump(per_sample_mje, f, indent=4)

            # =============== Mean per bone angular error after bone angle-based DTW alignment
            if m == "DTW-MBAE-BODY":
                aligned_grountruth_list, aligned_preds_list, _ = dtw_align(
                    gt_poses=groundtruth_list,
                    hypo_poses=preds_list,
                    criterion="bone_angle",
                    bones=BODY_BONES_FOR_ANGLES_SLRTP25,
                )
                eval[m], _ = mpbae(gt_poses=aligned_grountruth_list, hypo_poses=aligned_preds_list,
                                   bones=BONES_FOR_ANGLES_SLRTP25)

            if m == "DTW-MBAE-RH":
                aligned_grountruth_list, aligned_preds_list, _ = dtw_align(
                    gt_poses=groundtruth_list,
                    hypo_poses=preds_list,
                    criterion="bone_angle",
                    bones=RH_BONES_FOR_ANGLES_SLRTP25,
                )
                eval[m], _ = mpbae(gt_poses=aligned_grountruth_list, hypo_poses=aligned_preds_list,
                                   bones=RH_BONES_FOR_ANGLES_SLRTP25)

            if m == "DTW-MBAE-LH":
                aligned_grountruth_list, aligned_preds_list, _ = dtw_align(
                    gt_poses=groundtruth_list,
                    hypo_poses=preds_list,
                    criterion="bone_angle",
                    bones=LH_BONES_FOR_ANGLES_SLRTP25,
                )
                eval[m], _ = mpbae(gt_poses=aligned_grountruth_list, hypo_poses=aligned_preds_list,
                                   bones=LH_BONES_FOR_ANGLES_SLRTP25)

            # =============== Mean per bone orientation error after bone orientation-based DTW alignment
            if m == "DTW-MBOE-BODY":
                aligned_grountruth_list, aligned_preds_list, _ = dtw_align(
                    gt_poses=groundtruth_list,
                    hypo_poses=preds_list,
                    criterion="bone_orientation",
                    bones=BODY_BONES_FOR_ANGLES_SLRTP25,
                )
                eval[m], _ = mpboe(gt_poses=aligned_grountruth_list, hypo_poses=aligned_preds_list,
                                   bones=BONES_FOR_ANGLES_SLRTP25)

            if m == "DTW-MBOE-RH":
                aligned_grountruth_list, aligned_preds_list, _ = dtw_align(
                    gt_poses=groundtruth_list,
                    hypo_poses=preds_list,
                    criterion="bone_orientation",
                    bones=RH_BONES_FOR_ANGLES_SLRTP25,
                )
                eval[m], _ = mpboe(gt_poses=aligned_grountruth_list, hypo_poses=aligned_preds_list,
                                   bones=RH_BONES_FOR_ANGLES_SLRTP25)

            if m == "DTW-MBOE-LH":
                aligned_grountruth_list, aligned_preds_list, _ = dtw_align(
                    gt_poses=groundtruth_list,
                    hypo_poses=preds_list,
                    criterion="bone_orientation",
                    bones=LH_BONES_FOR_ANGLES_SLRTP25,
                )
                eval[m], _ = mpboe(gt_poses=aligned_grountruth_list, hypo_poses=aligned_preds_list,
                                   bones=LH_BONES_FOR_ANGLES_SLRTP25)

            # =============== Mean ratio of correct key-points after joint distance-based DTW alignment
            if m == "DTW-PCK":
                aligned_grountruth_list, aligned_preds_list, _ = dtw_align(
                    gt_poses=groundtruth_list,
                    hypo_poses=preds_list,
                    criterion="joint_distance",
                )
                eval[m], _ = pck(
                    gt_poses=aligned_grountruth_list,
                    hypo_poses=aligned_preds_list,
                    body_bones=BODY_BONES_FOR_ANGLES_SLRTP25,
                    hands_bones=HANDS_BONES_FOR_ANGLES_SLRTP25,
                    alpha=0.25,
                )

            # =============== Mean ratio of distance travelled by GT wrist over distance travelled by predicted wrist
            if m == "TotalDistance":
                eval[m] = total_distance(gt_poses=groundtruth_list, hypo_poses=preds_list)

                # =============== Back-translation metrics
            if m == "BLEU":
                assert text_preds is not None
                eval[m] = bleu(hypotheses=text_preds, references=text_groundtruth)
                print(eval[m])

            if m == "ROUGE-L":
                assert text_preds is not None
                eval[m], all_rouge = rouge(hypotheses=text_preds, references=text_groundtruth)
                if sample_vals_folder is not None:
                    assert len(all_rouge) == len(ex_ids)
                    per_sample_rouge = [
                        {"id": ex_ids[k], "rouge": all_rouge[k]}
                        for k in range(len(ex_ids))
                    ]
                    with open(f"{sample_vals_folder}/rouge.json", "w") as f:
                        json.dump(per_sample_rouge, f, indent=4)

            if m == "WER":
                assert text_preds is not None
                eval[m] = wer(hypotheses=text_preds, references=text_groundtruth)

            if m == "chrF":
                assert text_preds is not None
                eval[m] = chrf(hypotheses=text_preds, references=text_groundtruth)

            if m == "Likelihood":
                pass  # todo: implement

        return eval

    def evaluate_geo(self, preds, groundtruth, sample_vals_folder=None):
        return self.evaluate(preds, groundtruth, self.GEO_METRICS, sample_vals_folder)

    def evaluate_bt(self, preds, groundtruth, sample_vals_folder=None):
        return self.evaluate(preds, groundtruth, self.BT_METRICS, sample_vals_folder)

    def run_(
            self,
            preds_fpath: str,
            groundtruth_fpath: str,
            output_folder: str,
            name: str,
            metrics_type: str = "geo",  # "bt"
            store_sample_vals: bool = False,
            **prepro_steps,
    ):
        """
        Same as classical run() method but with an option to target geometric or back-translation metrics.
        """

        # 1) Loading data
        print("\nLoading data...")
        preds = self.load_predictions(preds_fpath)
        groundtruth = self.load_groundtruth(groundtruth_fpath)

        # 2) Optional preprocessing
        print("\nOptional preprocessing...")
        self.preprocessing(preds, groundtruth, **prepro_steps)

        # 3) Computing metrics
        sample_vals_folder = None
        if store_sample_vals:
            sample_vals_folder = f"{output_folder}/per_sample_results{name}"
            os.makedirs(sample_vals_folder, exist_ok=True)

        print("\nEvaluating...")
        eval = {}
        if metrics_type == "geo":
            eval["geometric_metrics"] = self.evaluate_geo(preds, groundtruth, sample_vals_folder=sample_vals_folder)
        elif metrics_type == "bt":
            eval["backtranslation_metrics"] = self.evaluate_bt(preds, groundtruth,
                                                               sample_vals_folder=sample_vals_folder)
        else:
            raise NotImplementedError(f"Unknown metrics type '{metrics_type}'")

        # 4) Saving as .json
        os.makedirs(output_folder, exist_ok=True)
        output_filepath = f"{output_folder}/{name}_{metrics_type}.json"
        print(f"\nSaving results to {output_filepath}")
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(eval, f, indent=4, sort_keys=True, ensure_ascii=False)

        return eval

    def run(
            self,
            preds_fpath: str,
            groundtruth_fpath: str,
            output_folder: str,
            name: str,
            **prepro_steps,
    ):

        # 1) Loading data
        print("\nLoading data...")
        preds = self.load_predictions(preds_fpath)
        groundtruth = self.load_groundtruth(groundtruth_fpath)

        # 2) Optional preprocessing
        print("\nOptional preprocessing...")
        self.preprocessing(preds, groundtruth, **prepro_steps)

        # 3) Computing metrics
        print("\nEvaluating...")
        eval = {
            "geometric_metrics": self.evaluate_geo(preds, groundtruth),
            "backtranslation_metrics": self.evaluate_bt(preds, groundtruth),
        }
        self.eval = eval.copy()

        # 4) Saving as .json
        os.makedirs(output_folder, exist_ok=True)
        output_filepath = f"{output_folder}/{name}.json"
        print(f"\nSaving results to {output_filepath}")
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(eval, f, indent=4, sort_keys=True, ensure_ascii=False)

        return eval
