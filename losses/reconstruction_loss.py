import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any

from ..data.skeletal_data import MediaPipe50KeyPoints, Skel178_face_ids, Skel178_mouth_ids


class ReconstructionLoss(nn.Module):
    losses = {

        # -- Standard losses
        "position": "position_error",
        "velocity": "velocity_error",
        "acceleration": "acceleration_error",

        # -- Per region losses
        # a/ position
        "hands_position": "hands_position_error",
        "torsoarms_position": "torsoarms_position_error",
        "rh_position": "rh_position_error",
        "lh_position": "lh_position_error",
        "face_position": "face_position_error",
        "mouth_position": "mouth_position_error",
        # b/ velocity
        "torsoarms_velocity": "torsoarms_velocity_error",
        "face_velocity": "face_velocity_error",
        "mouth_velocity": "mouth_velocity_error",

    }

    def __init__(
            self,
            list_losses: List[str] = None, scaling_factors: Dict[str, float] = None,
            losses_params: Dict[str, Any] = None,
            batch_norm: bool = True, target_pad: float = 0.0,
            apply_frames_weighting: str = None,
    ):
        super().__init__()
        if list_losses is None:
            list_losses = ["position"]
            if scaling_factors is None:
                scaling_factors = dict()
                scaling_factors["position"] = 1.
        self.list_losses = list_losses
        self.scaling_factors = scaling_factors
        self.batch_norm = batch_norm
        self.target_pad = target_pad
        self.losses_params = losses_params
        self.apply_frames_weighting = apply_frames_weighting
        if apply_frames_weighting is not None:
            print("Frames weighting strategy will be applied on position-based reconstruction loss,"
                  f" using the '{apply_frames_weighting}' method.")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, annealing_factors: dict[str, float] = None):
        # X is of shape (B, T, N, 3)
        B, T = targets.shape[:2]
        frame_mask = (targets.view(B, T, -1) != self.target_pad).any(dim=-1, keepdim=True).float()  # (B, T, 1)
        valid_timesteps = frame_mask.squeeze(-1)  # (B, T)
        gt_sequences_T = valid_timesteps.sum(dim=1)  # (B,)

        mask_joints = frame_mask.unsqueeze(-1).expand_as(targets)  # (B, T, Npts, 3)

        loss = 0.
        for l in self.list_losses:
            kwargs = {}
            if self.losses_params:
                kwargs = self.losses_params.get(l, {})
            if "position" in l:
                if self.apply_frames_weighting is not None:
                    kwargs["w"] = self.get_frames_weighting(x=targets, method=self.apply_frames_weighting)
                else:
                    kwargs["w"] = None
            l_val = getattr(self, self.losses[l])(
                x=preds, y=targets, ty=gt_sequences_T, mask=mask_joints,
                batch_norm=self.batch_norm,
                **kwargs,
            )
            af = 1.
            if annealing_factors:
                af = annealing_factors.get(l, 1.)
            loss += af * self.scaling_factors[l] * l_val

        return loss

    @staticmethod
    def get_frames_weighting(x: torch.Tensor, hands_ids: list[int] = None, method: str = "naive"):
        if method == "naive":
            # ---- 1. Hands only ----
            if hands_ids is None:
                hands_ids = MediaPipe50KeyPoints.ids_rh() + MediaPipe50KeyPoints.ids_lh()
            x_hands = x[:, :, hands_ids, :]  # (B, F, N_hands, 3)
            # ---- 2. Velocity ----
            v = x_hands[:, 1:] - x_hands[:, :-1]  # dX = X[t+1] - X[t]
            v = F.pad(v, (0, 0, 0, 0, 1, 0))  # one pad at the beginning (1st frame)
            v_mag = torch.norm(v, dim=-1).mean(dim=-1, keepdim=True)  # (B, F, 1) avg velocity magnitude per frame
            # ---- 3. Pose deviation (distance to mean pose) ----
            mean_pose = x_hands.mean(dim=1, keepdim=True)  # (B, 1, N_hands, 3) temporally avg hands poses
            d = torch.norm(x_hands - mean_pose, dim=-1).mean(dim=-1, keepdim=True)  # (B, F, 1) avg variation from mean
            d_vel = torch.abs(d[:, 1:] - d[:, :-1])
            d_vel = F.pad(d_vel, (0, 0, 1, 0))
            # ---- 4. Combine ----
            eps = 1e-4
            w = 0.1 + d_vel / (v_mag + eps)  # => higher w when higher variation from mean hands positions and lower velocity
            # ---- 5. Smooth ----
            w = w.permute(0, 2, 1)  # (B, 1, F)
            w = F.avg_pool1d(w, kernel_size=5, stride=1, padding=2)  # average pooling over last dimension (temporal)
            w = w.permute(0, 2, 1)  # (B, F, 1)
            # ---- 6. Normalize ----
            w = w / (w.mean(dim=1, keepdim=True) + eps)
            return w.unsqueeze(-1)  # (B, F, 1, 1)
        elif method == "topk":
            # ---- 1. Hands only ----
            if hands_ids is None:
                hands_ids = MediaPipe50KeyPoints.ids_rh() + MediaPipe50KeyPoints.ids_lh()
            x_hands = x[:, :, hands_ids, :]  # (B, F, N_hands, 3)
            # ---- 2. Velocity ----
            v = x_hands[:, 1:] - x_hands[:, :-1]  # dX = X[t+1] - X[t]
            v = F.pad(v, (0, 0, 0, 0, 1, 0))  # one pad at the beginning (1st frame)
            v_mag = torch.norm(v, dim=-1).mean(dim=-1, keepdim=True)  # (B, F, 1) avg velocity magnitude per frame
            # ---- 3. Pose deviation (distance to mean pose) ----
            mean_pose = x_hands.mean(dim=1, keepdim=True)  # (B, 1, N_hands, 3) temporally avg hands poses
            d = torch.norm(x_hands - mean_pose, dim=-1).mean(dim=-1, keepdim=True)  # (B, F, 1) avg variation from mean
            d_vel = torch.abs(d[:, 1:] - d[:, :-1])
            d_vel = F.pad(d_vel, (0, 0, 1, 0))
            # ---- 4. Combine ----
            eps = 1e-4
            w = d_vel / (v_mag + eps)  # => higher w when higher variation from mean hands positions and lower velocity
            # ---- 5. Smooth ----
            w = w.permute(0, 2, 1)  # (B, 1, F)
            w = F.avg_pool1d(w, kernel_size=5, stride=1, padding=2)  # average pooling over last dimension (temporal)
            w = w.permute(0, 2, 1)  # (B, F, 1)
            # ---- 6. Normalize ----
            w = w / (w.mean(dim=1, keepdim=True) + eps)
            k = int(0.2 * x.shape[1])  # top 20%
            w_flat = w.squeeze(-1)  # (B, F)
            threshold = torch.topk(w_flat, k, dim=1).values[:, -1:]  # (B, 1)
            mask = (w_flat >= threshold).float()  # (B, F)
            w = w * mask.unsqueeze(-1)  # (B, F, 1)
            w = w / (w.mean(dim=1, keepdim=True) + eps)
            return w.unsqueeze(-1)  # (B, F, 1, 1)
        else:
            raise NotImplementedError(f"Unknown method '{method}' for keyframes weights computation")


    @staticmethod
    def position_error(x, y, ty, mask, batch_norm: bool = False, loss: str = "mse", w: torch.Tensor = None):
        N = y.shape[2]  # Y shape=(B, T, Npts, 3)
        if loss == "mse":
            err = ((x - y) ** 2) * mask  # shape=(B, T, Npts, 3)
        elif loss == "l1":
            err = torch.abs(x - y) * mask  # shape=(B, T, Npts, 3)
        else:
            raise ValueError(f"Invalid loss type: {loss}")
        if w is not None:
            err = w * err
        err = err.sum(dim=(1, 2, 3)) / (N * ty)
        if batch_norm:
            err = err / y.shape[0]
        return err.sum()

    @staticmethod
    def velocity_error(x, y, ty, mask, batch_norm: bool = False, loss: str = "mse"):
        N = y.shape[2]  # Y shape=(B, T, Npts, 3)
        x_masked = x * mask
        y_masked = y * mask
        dx = x_masked[:, 1:] - x_masked[:, :-1]
        dy = y_masked[:, 1:] - y_masked[:, :-1]
        vel_frame_mask = mask.any(dim=(2, 3))  # (B, T)
        vel_frame_mask = vel_frame_mask[:, 1:]
        if loss == "mse":
            err = ((dx - dy) ** 2)  # shape=(B, T - 1, Npts, 3)
        elif loss == "l1":
            err = torch.abs(dx - dy)
        else:
            raise ValueError(f"Invalid loss type: {loss}")
        err = err * vel_frame_mask[:, :, None, None]
        err = err.sum(dim=(1, 2, 3)) / (N * (ty - 1))
        if batch_norm:
            err = err / y.shape[0]
        return err.sum()  # scalar

    @staticmethod
    def acceleration_error(x, y, ty, mask, batch_norm: bool = False):
        N = y.shape[2]  # Y shape=(B, T, Npts, 3)
        x_masked = x * mask
        y_masked = y * mask
        d2x = x_masked[:, 2:] - 2 * x_masked[:, 1:-1] + x_masked[:, :-2]
        d2y = y_masked[:, 2:] - 2 * y_masked[:, 1:-1] + y_masked[:, :-2]
        sq_err = ((d2x - d2y) ** 2)  # shape=(B, T - 2, Npts, 3)
        acc_frame_mask = mask.any(dim=(2, 3))  # (B, T)
        acc_frame_mask = acc_frame_mask[:, 2:]
        sq_err = sq_err * acc_frame_mask[:, :, None, None]
        mse = sq_err.sum(dim=(1, 2, 3)) / (N * (ty - 2))
        if batch_norm:
            mse = mse / y.shape[0]
        return mse.sum()  # scalar

    def Skel178Keypoints_region_error(
            self,
            x, y, ty, mask,
            batch_norm: bool = False,
            quantity: str = "position",  # or "velocity"
            region_name: str = None,
            region_ids: list[int] = None,
            center_at_id: int = None,  # id in the tensor of the extracted region (not in the full body)!!
            loss: str = "l1",
            w: torch.Tensor = None,
    ):
        """
        Helper general method.
        Position or velocity error loss on a specific region of the 178 key-points skeleton.
        """
        if region_name is not None:
            if region_name == "torsoarms":
                region_ids = MediaPipe50KeyPoints.ids_torsoarms()
            elif region_name == "rh":
                region_ids = MediaPipe50KeyPoints.ids_rh()
            elif region_name == "lh":
                region_ids = MediaPipe50KeyPoints.ids_lh()
            elif region_name == "face":
                region_ids = Skel178_face_ids()
            elif region_name == "mouth":
                region_ids = Skel178_mouth_ids()
            else:
                raise NotImplementedError(f"Invalid `region_name`: '{region_name}'")
        elif region_ids is None:
            raise ValueError(f"You must provide either `region_name` or `region_ids` arguments")

        x_reg = x[:, :, region_ids]
        y_reg = y[:, :, region_ids]
        mask_reg = mask[:, :, region_ids]

        if center_at_id is not None:
            x_reg = x_reg - x_reg[:, :, [center_at_id]]
            y_reg = y_reg - y_reg[:, :, [center_at_id]]

        if quantity == "position":
            return self.position_error(x_reg, y_reg, ty, mask_reg, batch_norm, loss, w)
        elif quantity == "velocity":
            return self.velocity_error(x_reg, y_reg, ty, mask_reg, batch_norm, loss)
        else:
            raise NotImplementedError(f"Unknown quantity '{quantity}'")

    # -- Regions position error
    def torsoarms_position_error(self, x, y, ty, mask, batch_norm, loss, w):
        return self.Skel178Keypoints_region_error(
            x, y, ty, mask, batch_norm,
            region_name="torsoarms",
            center_at_id=None,
            loss=loss,
            w=w,
        )

    def rh_position_error(self, x, y, ty, mask, batch_norm, loss, w):
        return self.Skel178Keypoints_region_error(
            x, y, ty, mask, batch_norm,
            region_name="rh",
            center_at_id=0,
            loss=loss,
            w=w,
        )

    def lh_position_error(self, x, y, ty, mask, batch_norm, loss, w):
        return self.Skel178Keypoints_region_error(
            x, y, ty, mask, batch_norm,
            region_name="lh",
            center_at_id=0,
            loss=loss,
            w=w,
        )

    def face_position_error(self, x, y, ty, mask, batch_norm, loss, w):
        return self.Skel178Keypoints_region_error(
            x, y, ty, mask, batch_norm,
            region_name="face",
            center_at_id=None,
            loss=loss,
            w=w,
        )

    def mouth_position_error(self, x, y, ty, mask, batch_norm, loss, w):
        return self.Skel178Keypoints_region_error(
            x, y, ty, mask, batch_norm,
            region_name="mouth",
            center_at_id=None,
            loss=loss,
            w=w,
        )

    # -- Regions velocity error
    def torsoarms_velocity_error(self, x, y, ty, mask, batch_norm, loss):
        return self.Skel178Keypoints_region_error(
            x, y, ty, mask, batch_norm,
            quantity="velocity",
            region_name="torsoarms",
            center_at_id=None,
            loss=loss,
        )

    def face_velocity_error(self, x, y, ty, mask, batch_norm, loss):
        return self.Skel178Keypoints_region_error(
            x, y, ty, mask, batch_norm,
            quantity="velocity",
            region_name="face",
            center_at_id=None,
            loss=loss,
        )

    def mouth_velocity_error(self, x, y, ty, mask, batch_norm, loss):
        return self.Skel178Keypoints_region_error(
            x, y, ty, mask, batch_norm,
            quantity="velocity",
            region_name="mouth",
            center_at_id=None,
            loss=loss,
        )

    # -- Both hands position error
    # (might be unnecessary now that we have specific RH / LH losses)
    @staticmethod
    def hands_position_error(
            x, y, ty, mask, batch_norm: bool = False,
            hands_ids: Tuple[List[int], List[int]] = None, wrists_ids: Tuple[int, int] = None,
            weigthing_frames_method: str = None,
            loss: str = "mse",  # 'l1'
    ):
        if hands_ids is None:
            hands_ids = ([i for i in range(8, 29)], [i for i in range(29, 50)])
            wrists_ids = (8, 29)
        x_hands = torch.cat(
            (x[:, :, hands_ids[0]] - x[:, :, [wrists_ids[0]]],  # RH
             x[:, :, hands_ids[1]] - x[:, :, [wrists_ids[1]]]),  # LH
            dim=2
        )
        y_hands = torch.cat(
            (y[:, :, hands_ids[0]] - y[:, :, [wrists_ids[0]]],
             y[:, :, hands_ids[1]] - y[:, :, [wrists_ids[1]]]),
            dim=2
        )
        w = 1
        if weigthing_frames_method == "fingers_directions":
            pass
        if weigthing_frames_method == "points_dispersion":
            disp_per_frameR = y_hands[:, :, :21].var(dim=2).sum(dim=-1)  # (B, T)
            disp_per_frameL = y_hands[:, :, 21:].var(dim=2).sum(dim=-1)  # (B, T)
            w = F.softmax(disp_per_frameR + disp_per_frameL, dim=1)  # (B, T)
            w = w[:, :, None, None]
        mask_hands = mask[:, :, hands_ids[0] + hands_ids[1]]
        N = len(hands_ids[0]) + len(hands_ids[1])  # Y shape=(B, T, Npts_hands, 3)
        if loss == 'mse':
            err = w * ((x_hands - y_hands) ** 2) * mask_hands  # shape=(B, T, Npts_hands, 3)
        if loss == 'l1':
            err = w * torch.abs(x_hands - y_hands) * mask_hands  # shape=(B, T, Npts_hands, 3)
        err = err.sum(dim=(1, 2, 3)) / (N * ty)
        if batch_norm:
            err = err / y.shape[0]
        return err.sum()
