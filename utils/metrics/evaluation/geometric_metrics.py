import sys

import torch
from typing import List, Tuple
from tqdm import tqdm
from fastdtw import fastdtw as dtw
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
from pose_evaluation.evaluation.create_metrics import construct_metric
from utils.metrics.mp178_to_body import MP178_to_Body
from utils.metrics.slrtp_challenge_2025_evaluation.metrics import pose_distance


def dtwp(gt_poses: List[torch.Tensor], hypo_poses: List[torch.Tensor], fps=25) -> Tuple[float, List[float]]:
    """Based on https://github.com/sign-language-processing/pose-evaluation (see: https://arxiv.org/abs/2510.07453)"""
    assert len(gt_poses) == len(hypo_poses)
    
    DTWp = construct_metric(
        distance_measure=DTWDTAIImplementationDistanceMeasure(name="dtaiDTWAggregatedDistanceMeasureFast", use_fast=True),
        default_distance=0.0,
        trim_meaningless_frames=True,
        normalize=False,
        sequence_alignment="dtw",
        keypoint_selection="hands", # keep only hand keypoints for all poses
        masked_fill_value=10.0, # fill masked values with 10.0
        fps=None, # don't interpolate fps
        name=None, # autogenerate name
    )
    
    dtwps = []
    for gt, hypo in tqdm(zip(gt_poses, hypo_poses), total=len(gt_poses), desc="Computing DTWp for each sequence..."):
        gt_pose = MP178_to_Body(gt.unsqueeze(1), fps)
        pred_pose = MP178_to_Body(hypo.unsqueeze(1), fps)
        dtwps.append(DTWp(pred_pose, gt_pose))
    
    return torch.tensor(dtwps).mean().item(), dtwps


def dtw_align(
    gt_poses: List[torch.Tensor], 
    hypo_poses: List[torch.Tensor], 
    criterion: str = "joint_distance",
    bones: List[Tuple[int, int]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
    assert len(gt_poses) == len(hypo_poses)
    
    if criterion == "joint_distance":
        def euclidean_distance(x, y):
            """x and y are of shape (N_joints * 3,)"""
            x = torch.tensor(x)
            y = torch.tensor(y)
            return torch.sqrt(torch.sum((x - y) ** 2))
        
        dist_func = euclidean_distance
    
    elif criterion == "bone_angle":
        # def angle_error(x, y):
        #     """x and y are of shape (N_joints * 3,)"""
        #     x = torch.tensor(x).view(-1, 3)  # shape=(N_joints, 3)
        #     y = torch.tensor(y).view(-1, 3)
        #     errors = []
        #     for p, c in bones:
        #         # vectors along the bones
        #         v_x = x[c] - x[p]  # shape=(3,)
        #         v_y = y[c] - y[p]
        #         # cosine of angle between vectors
        #         cos_angle = (v_x * v_y).sum() / (v_x.norm() * v_y.norm() + 1e-8)
        #         cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # numerical stability
        #         errors.append(torch.acos(cos_angle))
        #     return torch.tensor(errors).sum()
        
        # ======= faster new version
        def angle_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            Compute total angular error between two poses (vectorized).
            x, y: shape (N_joints * 3,)
            Returns: scalar tensor (sum of angular errors)
            """
            # Ensure tensor shape (N_joints, 3)
            x = torch.tensor(x).view(-1, 3)
            y = torch.tensor(y).view(-1, 3)
            # Extract all bone endpoints at once
            parents = torch.tensor([p for p, _ in bones], device=x.device)
            children = torch.tensor([c for _, c in bones], device=x.device)
            # Compute bone vectors in batch
            vx = x[children] - x[parents]   # (Nbones, 3)
            vy = y[children] - y[parents]   # (Nbones, 3)
            # Cosine of angles (vectorized)
            dot = (vx * vy).sum(dim=-1)
            norm_prod = vx.norm(dim=-1) * vy.norm(dim=-1)
            cos_angle = dot / (norm_prod + 1e-8)
            # Clamp for numerical stability
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            # Sum angular errors
            return torch.acos(cos_angle).sum()

        dist_func = angle_error

    elif criterion == "bone_orientation":
        def orientation_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            Compute total orientation error between two poses (vectorized).
            I.e, given the lists of bone vectors Bx = (bx1, ..., bxM) and By = (by1, ..., byM),
            computes: Sum_i |bxi - byi|_2
            x, y: shape (N_joints * 3,)
            Returns: scalar tensor (sum of bone orientation errors)
            """
            # Ensure tensor shape (N_joints, 3)
            x = torch.tensor(x).view(-1, 3)
            y = torch.tensor(y).view(-1, 3)
            # Extract all bone endpoints at once
            parents = torch.tensor([p for p, _ in bones], device=x.device)
            children = torch.tensor([c for _, c in bones], device=x.device)
            # Compute bone vectors in batch
            vx = x[children] - x[parents]  # (Nbones, 3)
            vy = y[children] - y[parents]  # (Nbones, 3)
            # Compute orientation error (reduction=sum)
            vx_norm = vx / (torch.norm(vx, dim=-1, keepdim=True) + 1e-8)
            vy_norm = vy / (torch.norm(vy, dim=-1, keepdim=True) + 1e-8)
            return torch.sqrt(((vx_norm - vy_norm) ** 2).sum(dim=-1)).sum()

        dist_func = orientation_error

    else:
        raise ValueError("Unknown input criterion: must be either 'joint_distance' or 'bone_angle'")
    
    def dtw_align_data(a: list, b: list, dist_fn):
        align_a = []
        align_b = []
        distances = []
        for _a, _b in tqdm(zip(a, b), total=len(a), desc=f"DTW alignment using '{criterion}'..."):
            #  skip blank sequences
            if _a is None or _b is None:
                continue
            if len(_a) == 0 or len(_b) == 0:
                continue
            dist, path = dtw(_a.flatten(1, -1), _b.flatten(1, -1), dist=dist_fn)  # NB: dist is between _a[t] and _b[t'] (not _a and _b)
            a_path, b_path = zip(*path)
            _a = _a[list(a_path)]
            _b = _b[list(b_path)]
            assert _a.shape == _b.shape
            align_a.append(_a)
            align_b.append(_b)
            distances.append(dist)
        return align_a, align_b, distances

    gt_poses_aligned, hypo_poses_aligned, dtw_distances = dtw_align_data(gt_poses, hypo_poses, dist_func)
    
    return gt_poses_aligned, hypo_poses_aligned, torch.tensor(dtw_distances).mean().item()


def mpje(gt_poses: List[torch.Tensor], hypo_poses: List[torch.Tensor]) -> Tuple[float, List[float]]:
    """
    Adapted from https://github.com/walsharry/SLRTP-Sign-Production-Evaluation
    Calculate the Mean Per Joint Position Error (MPJPE) between ground truth and hypothesis poses.
    
    NB: gt_poses and hypo_poses must contain the same number of sequences, and in the SAME order (matching).

    Args:
    - gt_poses (List[torch.Tensor]): List of sequences of ground truth poses with shape (T_seq, num_joints, 3).
    - hypo_poses (List[torch.Tensor]): List of sequences of hypothesis poses with shape (T_seq, num_joints, 3).

    Returns:
    - Mean Per Joint Position Error (float) averaged over sequences
    - MPJE of each sequence (list[float])
    """
    assert (isinstance(gt_poses, list) and isinstance(hypo_poses, list))
    
    N_seq = len(gt_poses)

    def cal_mpjpe_per_sequence(list_sequences_a: List[torch.Tensor], list_sequences_b: List[torch.Tensor]):
        # Safeguards checks
        assert len(list_sequences_a) == len(list_sequences_b), "Expected same number of GT and predicted sequences!"

        # Calculate mean Euclidean distance over time and joints between each (GT, prediction) pair of sequences
        mpjpe_per_seq = []
        for seq_a, seq_b in tqdm(
                zip(list_sequences_a, list_sequences_b),
                total=len(list_sequences_a), desc="Computing MPJPE for each sequence..."
        ):
            assert seq_a.shape == seq_b.shape, "GT and predicted sequence shapes must match!"  # shape=(T_seq, N_joints, 3)
            mpjpe_per_seq.append(torch.mean(torch.norm(seq_a - seq_b, dim=2)).item())

        return mpjpe_per_seq

    def cal_mpjpe(a, b):
        # Check if the input tensors have the same shape
        assert (
            a.shape == b.shape
        ), "Ground truth and hypothesis poses must have the same shape."

        # Calculate the Euclidean distance between corresponding joints
        joint_distances = torch.norm(a - b, dim=2)

        # Calculate the mean over all joints and samples
        mean_mpjpe = torch.mean(joint_distances)
        return mean_mpjpe

    list_mean_joint_error_per_seq = cal_mpjpe_per_sequence(gt_poses, hypo_poses)
    gt_poses = torch.cat(gt_poses)  # shape=(N_seq * T_seq, N_joints, 3)
    hypo_poses = torch.cat(hypo_poses)

    return cal_mpjpe(gt_poses, hypo_poses).item(), list_mean_joint_error_per_seq


def mpbae(gt_poses: List[torch.Tensor], hypo_poses: List[torch.Tensor], bones: List[Tuple[int, int]] = None) -> Tuple[float, List[float]]:
    """
    Calculate the Mean Per Bone Angle Error (MPBAE) between ground truth and hypothesis poses (for the given region).
    The result is given in degrees!
    
    NB: gt_poses and hypo_poses must contain the same number of sequences, and in the SAME order (matching).

    Args:
    - gt_poses (List[torch.Tensor]): List of sequences of ground truth poses with shape (T_seq, num_joints, 3).
    - hypo_poses (List[torch.Tensor]): List of sequences of hypothesis poses with shape (T_seq, num_joints, 3).
    - bones (List[Tuple[int, int]]): List of bones as tuples of the form (i_parent, i_child)

    Returns:
    - Mean Per Bone Angle Error (float) averaged over sequences
    - MPBAE of each sequence (list[float])
    """
    all_errors = []
    per_sequence_mean_error = []
    
    for gt, hypo in tqdm(zip(gt_poses, hypo_poses), total=len(gt_poses), desc="Computing MBAE for each sequence..."):
        # compute bone vectors
        gt_bones = torch.stack([gt[:, c] - gt[:, p] for p, c in bones], dim=1)  # (T, Nbones, 3)
        hypo_bones = torch.stack([hypo[:, c] - hypo[:, p] for p, c in bones], dim=1)  # (T, Nbones, 3)
        # cosine similarity
        dot = (gt_bones * hypo_bones).sum(-1)  # (T, Nbones)
        norms = gt_bones.norm(dim=-1) * hypo_bones.norm(dim=-1)  # (T, Nbones)
        cos_angle = torch.clamp(dot / (norms + 1e-8), -1.0, 1.0)
        errors = torch.acos(cos_angle)  # (T, Nbones)
        
        to_degree = 180/torch.pi
        per_sequence_mean_error.append(errors.mean().item() * to_degree)  # avering over time and bones
        all_errors.append(errors * to_degree)
        
    all_errors = torch.cat(all_errors, dim=0)  # (sum_T, Nbones)
    return all_errors.mean().item(), per_sequence_mean_error


def mpboe(gt_poses: List[torch.Tensor], hypo_poses: List[torch.Tensor], bones: List[Tuple[int, int]] = None) -> Tuple[
    float, List[float]]:
    """
    Calculate the Mean Per Bone Orientation Error (MPBOE) between ground truth
    and hypothesis poses (for the given region).

    NB: gt_poses and hypo_poses must contain the same number of sequences, and in the SAME order (matching).

    Args:
    - gt_poses (List[torch.Tensor]): List of sequences of ground truth poses with shape (T_seq, num_joints, 3).
    - hypo_poses (List[torch.Tensor]): List of sequences of hypothesis poses with shape (T_seq, num_joints, 3).
    - bones (List[Tuple[int, int]]): List of bones as tuples of the form (i_parent, i_child)

    Returns:
    - Mean Per Bone Orientation Error (float) averaged over sequences
    - MPBOE of each sequence (list[float])
    """
    all_errors = []
    per_sequence_mean_error = []

    for gt, hypo in tqdm(zip(gt_poses, hypo_poses), total=len(gt_poses), desc="Computing MBOE for each sequence..."):
        # compute bone vectors
        gt_bones = torch.stack([gt[:, c] - gt[:, p] for p, c in bones], dim=1)  # (T, Nbones, 3)
        hypo_bones = torch.stack([hypo[:, c] - hypo[:, p] for p, c in bones], dim=1)  # (T, Nbones, 3)
        # compute mean orientation error per bone
        gt_bones_norm = gt_bones / (torch.norm(gt_bones, dim=-1, keepdim=True) + 1e-8)
        hypo_bones_norm = hypo_bones / (torch.norm(hypo_bones, dim=-1, keepdim=True) + 1e-8)
        errors = torch.sqrt(((gt_bones_norm - hypo_bones_norm) ** 2).sum(dim=-1)).mean(dim=1)
        # average over sequence (temporal mean)
        per_sequence_mean_error.append(errors.mean().item())
        all_errors.append(errors)

    all_errors = torch.cat(all_errors, dim=0)  # (sum_T, Nbones)
    return all_errors.mean().item(), per_sequence_mean_error


def pck(
    gt_poses: List[torch.Tensor],
    hypo_poses: List[torch.Tensor],
    body_bones: List[Tuple[int, int]], 
    hands_bones: List[Tuple[int, int]], 
    alpha: float = 0.25
    ):
    """
    Calculate the "Probability" of Correct Keypoint (PCK) between ground truth and hypothesis poses.
    
    NB: gt_poses and hypo_poses must contain the same number of sequences, and in the SAME order (correspondance).

    Args:
    - gt_poses (List[torch.Tensor]): List of sequences of ground truth poses with shape (T_seq, num_joints, 3).
    - hypo_poses (List[torch.Tensor]): List of sequences of hypothesis poses with shape (T_seq, num_joints, 3).
    - body_bones (List[int]): arms/torso skeletal bones connections ids
    - hands_bones (List[int]): hands skeletal bones connections ids
    - alpha (float): proportion of the median length of region bones to define neigborhood radius (default is 0.25)
                     E.g. if the median length of hands bones (fingers, etc) is 0.05, we will say that a
                     predicted hand joint J_pred is correct if it falls in the ball B(J_gt, 0.05 * 25%) 
                     (i.e. of center=ground truth joint and radius=25% * 0.05)

    Returns:
    - Probability of Correct Keypoint (float) averaged over sequences
    - PCK of each sequence (list[float])
    """
    per_sequence_pck = []
    
    def list_tuple_to_unique_ids(list_tuple):
        return list(set([a for a, _ in list_tuple] + [b for _, b in list_tuple]))
    
    body_ids = list_tuple_to_unique_ids(body_bones)
    hands_ids = list_tuple_to_unique_ids(hands_bones)
    
    for gt, hypo in tqdm(zip(gt_poses, hypo_poses), total=len(gt_poses), desc="Computing PCK for each sequence..."):
        # reminder: shape=(T_seq, Njoints, 3)
        
        # 1) Get body (arms) median bone length & hands median bone length ! FROM GROUND TRUTH !
        body_vecs  = torch.stack([gt[:, c] - gt[:, p] for p, c in body_bones], dim=0)  # (Nbones_body, T, 3)
        hands_vecs = torch.stack([gt[:, c] - gt[:, p] for p, c in hands_bones], dim=0)  # (Nbones_hands, T, 3)
        median_body_len  = body_vecs.norm(dim=-1).median().item()  # median over bones and time
        median_hands_len = hands_vecs.norm(dim=-1).median().item()
        
        # 2) Get number of correct body and hands joints
        correct_body_joints = (torch.norm(hypo[:, body_ids] - gt[:, body_ids], dim=-1) <= alpha * median_body_len)  # shape=(T_seq, Njoints_body)
        correct_hands_joints = (torch.norm(hypo[:, hands_ids] - gt[:, hands_ids], dim=-1) <= alpha * median_hands_len)  # shape=(T_seq, Njoints_hands)
        
        pck_body = correct_body_joints.float().mean()
        pck_hands = correct_hands_joints.float().mean()
        seq_pck = (pck_body + pck_hands) / 2
        
        # 3) Add per sequence PCK to the list
        per_sequence_pck.append(seq_pck.item())
    
    return torch.tensor(per_sequence_pck).mean().item(), per_sequence_pck


def total_distance(gt_poses: List[torch.Tensor], hypo_poses: List[torch.Tensor]) -> float:
    return pose_distance(hyps=hypo_poses, gt_pose=gt_poses)
