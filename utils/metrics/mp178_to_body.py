import torch
from pose_format.torch.pose_body import TorchPoseBody
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format import Pose
from pose_format.pose_header import (
    PoseHeader,
    PoseHeaderDimensions,
    PoseHeaderComponent,
)


# ===================================== Skel info ========================================
# -------------------------------------- General -----------------------------------------
IDS = {
    # ------------- TORSO -------------
    'RIGHT_SHOULDER': 0, 'RIGHT_ELBOW': 1, 'RIGHT_WRIST': 2,
    'LEFT_SHOULDER': 3, 'LEFT_ELBOW': 4, 'LEFT_WRIST': 5,
    'RIGHT_HIP': 6, 'LEFT_HIP': 7,

    # ------------- RH -------------
    'RHWrist': 8, 'RThumb1CMC': 9, 'RThumb2Knuckles': 10, 'RThumb3IP': 11, 'RThumb4FingerTip': 12,
    'RIndex1Knuckles': 13, 'RIndex2PIP': 14, 'RIndex3DIP': 15, 'RIndex4FingerTip': 16,
    'RMiddle1Knuckles': 17, 'RMiddle2PIP': 18, 'RMiddle3DIP': 19, 'RMiddle4FingerTip': 20,
    'RRing1Knuckles': 21, 'RRing2PIP': 22, 'RRing3DIP': 23, 'RRing4FingerTip': 24,
    'RPinky1Knuckles': 25, 'RPinky2PIP': 26, 'RPinky3DIP': 27, 'RPinky4FingerTip': 28,

    # ------------- LH -------------
    'LHWrist': 29, 'LThumb1CMC': 30, 'LThumb2Knuckles': 31, 'LThumb3IP': 32, 'LThumb4FingerTip': 33,
    'LIndex1Knuckles': 34, 'LIndex2PIP': 35, 'LIndex3DIP': 36, 'LIndex4FingerTip': 37,
    'LMiddle1Knuckles': 38, 'LMiddle2PIP': 39, 'LMiddle3DIP': 40, 'LMiddle4FingerTip': 41,
    'LRing1Knuckles': 42, 'LRing2PIP': 43, 'LRing3DIP': 44, 'LRing4FingerTip': 45,
    'LPinky1Knuckles': 46, 'LPinky2PIP': 47, 'LPinky3DIP': 48, 'LPinky4FingerTip': 49,

    # ------------- FACE -------------
    **{f"face{i}": i + 50 for i in range(128)}
}
# ---------------------------------------------------------------------------------------
# ------------------------------------- Point Names -------------------------------------
BODY_PTS = [
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "RIGHT_HIP", "LEFT_HIP"
]
RH_PTS = [
    "RHWrist",
    "RThumb1CMC", "RThumb2Knuckles", "RThumb3IP", "RThumb4FingerTip",
    "RIndex1Knuckles", "RIndex2PIP", "RIndex3DIP", "RIndex4FingerTip",
    "RMiddle1Knuckles", "RMiddle2PIP", "RMiddle3DIP", "RMiddle4FingerTip",
    "RRing1Knuckles", "RRing2PIP", "RRing3DIP", "RRing4FingerTip",
    "RPinky1Knuckles", "RPinky2PIP", "RPinky3DIP", "RPinky4FingerTip"
]
LH_PTS = [
    "LHWrist",
    "LThumb1CMC", "LThumb2Knuckles", "LThumb3IP", "LThumb4FingerTip",
    "LIndex1Knuckles", "LIndex2PIP", "LIndex3DIP", "LIndex4FingerTip",
    "LMiddle1Knuckles", "LMiddle2PIP", "LMiddle3DIP", "LMiddle4FingerTip",
    "LRing1Knuckles", "LRing2PIP", "LRing3DIP", "LRing4FingerTip",
    "LPinky1Knuckles", "LPinky2PIP", "LPinky3DIP", "LPinky4FingerTip"
]
FACE_PTS = [f"face{i}" for i in range(128)]
# -----------------------------------------------------------------------------------
# ----------------------------------- Connexions ------------------------------------
BODY_CON = [
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "LEFT_SHOULDER"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_HIP", "LEFT_HIP"),
]
RH_CON = [
    ("RIGHT_WRIST", "RHWrist"),
    ("RHWrist", "RThumb1CMC"),
    ("RThumb1CMC", "RThumb2Knuckles"),
    ("RThumb2Knuckles", "RThumb3IP"),
    ("RThumb3IP", "RThumb4FingerTip"),

    ("RHWrist", "RIndex1Knuckles"),
    ("RIndex1Knuckles", "RIndex2PIP"),
    ("RIndex2PIP", "RIndex3DIP"),
    ("RIndex3DIP", "RIndex4FingerTip"),

    ("RIndex1Knuckles", "RMiddle1Knuckles"),
    ("RMiddle1Knuckles", "RMiddle2PIP"),
    ("RMiddle2PIP", "RMiddle3DIP"),
    ("RMiddle3DIP", "RMiddle4FingerTip"),

    ("RMiddle1Knuckles", "RRing1Knuckles"),
    ("RRing1Knuckles", "RRing2PIP"),
    ("RRing2PIP", "RRing3DIP"),
    ("RRing3DIP", "RRing4FingerTip"),

    ("RRing1Knuckles", "RPinky1Knuckles"),
    ("RPinky1Knuckles", "RPinky2PIP"),
    ("RPinky2PIP", "RPinky3DIP"),
    ("RPinky3DIP", "RPinky4FingerTip"),

    ("RPinky1Knuckles", "RHWrist"),
]
LH_CON = [
    ("LEFT_WRIST", "LHWrist"),
    ("LHWrist", "LThumb1CMC"),
    ("LThumb1CMC", "LThumb2Knuckles"),
    ("LThumb2Knuckles", "LThumb3IP"),
    ("LThumb3IP", "LThumb4FingerTip"),

    ("LHWrist", "LIndex1Knuckles"),
    ("LIndex1Knuckles", "LIndex2PIP"),
    ("LIndex2PIP", "LIndex3DIP"),
    ("LIndex3DIP", "LIndex4FingerTip"),

    ("LIndex1Knuckles", "LMiddle1Knuckles"),
    ("LMiddle1Knuckles", "LMiddle2PIP"),
    ("LMiddle2PIP", "LMiddle3DIP"),
    ("LMiddle3DIP", "LMiddle4FingerTip"),

    ("LMiddle1Knuckles", "LRing1Knuckles"),
    ("LRing1Knuckles", "LRing2PIP"),
    ("LRing2PIP", "LRing3DIP"),
    ("LRing3DIP", "LRing4FingerTip"),

    ("LRing1Knuckles", "LPinky1Knuckles"),
    ("LPinky1Knuckles", "LPinky2PIP"),
    ("LPinky2PIP", "LPinky3DIP"),
    ("LPinky3DIP", "LPinky4FingerTip"),

    ("LPinky1Knuckles", "LHWrist"),
]
FACE_CON = []

BODY_CON_IDS = [(IDS[a], IDS[b]) for (a, b) in BODY_CON]
RH_CON_IDS = [(IDS[a], IDS[b]) for (a, b) in RH_CON]
LH_CON_IDS = [(IDS[a], IDS[b]) for (a, b) in LH_CON]
FACE_CON_IDS = []
# -----------------------------------------------------------------------------------


def MP178_to_Body(data: torch.Tensor, fps: float = 25.) -> Pose:
    """
    Convert a torch.Tensor (T, C, 178, 3) representing MediaPipe-style skeletal data
    into a pose-format Pose object with a TorchPoseBody and corresponding header.

    Args:
        data (torch.Tensor): Pose data of shape (T, C, 178, 3)
        fps (float): Frames per second of the sequence

    Returns:
        Pose: pose-format Pose object ready for saving or processing
    """

    if data.ndim != 4 or data.shape[-2:] != (178, 3):
        raise ValueError(f"Expected shape (T, C, 178, 3), got {tuple(data.shape)}")  # C is number of people

    # Confidence (optional, all ones)
    confidence = torch.ones_like(data[..., 0])  # (T, C, J)

    # Create TorchPoseBody
    # body = TorchPoseBody(fps=fps, data=data, confidence=confidence)
    body = NumPyPoseBody(fps=fps, data=data.numpy(), confidence=confidence.numpy())  # better numpy because some functions in pose-evaluation are not made BodyPose with torch tensors

    # Header Dimensions
    dimensions = PoseHeaderDimensions(width=500, height=500, depth=0)
    
    # Header Components
    black_rgb = (255, 255, 255)  # cf. https://rgbcolorpicker.com/
    red_rgb = (214, 33, 33)
    blue_rgb = (42, 32, 227)
    green_rgb = (25, 153, 30)

    body_component = PoseHeaderComponent(
        name="POSE_LANDMARKS",
        points=BODY_PTS,
        limbs=BODY_CON_IDS,
        colors=[black_rgb],
        point_format="XYZC",
    )
    righ_hand_component = PoseHeaderComponent(
        name="RIGHT_HAND_LANDMARKS",
        points=RH_PTS,
        limbs=RH_CON_IDS,
        colors=[green_rgb],
        point_format="XYZC",
    )
    left_hand_component = PoseHeaderComponent(
        name="LEFT_HAND_LANDMARKS",
        points=LH_PTS,
        limbs=LH_CON_IDS,
        colors=[blue_rgb],
        point_format="XYZC",
    )
    face_component = PoseHeaderComponent(
        name="FACE_LANDMARKS",
        points=FACE_PTS,
        limbs=FACE_CON_IDS,
        colors=[red_rgb],
        point_format="XYZC",
    )
    components = [
        body_component,
        righ_hand_component,
        left_hand_component,
        # face_component,
    ]

    header = PoseHeader(
        version=1.0,  # default
        dimensions=dimensions,
        components=components,
    )

    # Combine into Pose
    pose = Pose(header=header, body=body)
    return pose
