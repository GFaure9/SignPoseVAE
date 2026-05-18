"""
Define skeletal structures
"""

import numpy as np


# For face regions cf. `order_128` dictionary in:
# https://github.com/walsharry/SLRTP-Sign-Production-Evaluation/blob/main/make_128_face_from_478.py

def Skel178_face_ids():
    return list(range(50, 178))

def Skel178_mouth_ids():
    """
    Cf. 'mouth_outline' and 'mouth_middle'
    """
    in_face128_alone = [
        # -- mouth outline
        30, 37, 36, 113, 112,
        32, 31, 75, 74, 56,
        55, 114, 123, 91, 92,
        78, 79, 118, 63, 29,
        # -- mouth middle
        45, 80, 5, 4, 34,
        1, 0, 12, 11, 117,
        116, 122, 125, 90, 8,
        9, 73, 115, 41, 42,
    ]
    face_offset_in_178skel = 50
    return (np.array(in_face128_alone) + face_offset_in_178skel).tolist()


class MediaPipe50KeyPoints:
    # cf. info at https://github.com/walsharry/SLRTP-Sign-Production-Evaluation

    @staticmethod
    def connections_torsoarms():
        # ==== in TORSO/ARMS indexation!!
        return [
            (0, 1),  # RShoulder -> RElbow
            (1, 2),  # RElbow -> RWrist

            (3, 4),  # LShoulder -> LElbow
            (4, 5),  # LElbow -> LWrist

            (0, 3),  # RShoulder -> LShoulder
            (0, 6),  # RShoulder -> RHip
            (3, 7),  # LShoulder -> LHip
            (6, 7),  # RHip -> LHip

            (2, 8),  # RWrist
            (5, 9),  # LWrist
        ]

    @staticmethod
    def connections_rh():
        # ==== in RH indexation!!
        return [
            (0, 1),  # RHWrist -> RThumb1CMC
            (1, 2),  # RThumb1CMC -> RThumb2Knuckles
            (2, 3),  # RThumb2Knuckles -> RThumb3IP
            (3, 4),  # RThumb3IP -> RThumb4FingerTip

            (0, 5),  # RHWrist -> RIndex1Knuckles
            (5, 6),  # RIndex1Knuckles -> RIndex2PIP
            (6, 7),  # RIndex2PIP -> RIndex3DIP
            (7, 8),  # RIndex3DIP -> RIndex4FingerTip

            (0, 9),  # RHWrist -> RMiddle1Knuckles
            (9, 10),  # RMiddle1Knuckles -> RMiddle2PIP
            (10, 11),  # RMiddle2PIP -> RMiddle3DIP
            (11, 12),  # RMiddle3DIP -> RMiddle4FingerTip

            (0, 13),  # RHWrist -> RRing1Knuckles
            (13, 14),  # RRing1Knuckles -> RRing2PIP
            (14, 15),  # RRing2PIP -> RRing3DIP
            (15, 16),  # RRing3DIP -> RRing4FingerTip

            (0, 17),  # RHWrist -> RPinky1Knuckles
            (17, 18),  # RPinky1Knuckles -> RPinky2PIP
            (18, 19),  # RPinky2PIP -> RPinky3DIP
            (19, 20),  # RPinky3DIP -> RPinky4FingerTip
        ]

    @staticmethod
    def connections_lh():
        # ==== in LH indexation!!
        return [
            (0, 1),  # LHWrist -> LThumb1CMC
            (1, 2),  # LThumb1CMC -> LThumb2Knuckles
            (2, 3),  # LThumb2Knuckles -> LThumb3IP
            (3, 4),  # LThumb3IP -> LThumb4FingerTip

            (0, 5),  # LHWrist -> LIndex1Knuckles
            (5, 6),  # LIndex1Knuckles -> LIndex2PIP
            (6, 7),  # LIndex2PIP -> LIndex3DIP
            (7, 8),  # LIndex3DIP -> LIndex4FingerTip

            (0, 9),  # LHWrist -> LMiddle1Knuckles
            (9, 10),  # LMiddle1Knuckles -> LMiddle2PIP
            (10, 11),  # LMiddle2PIP -> LMiddle3DIP
            (11, 12),  # LMiddle3DIP -> LMiddle4FingerTip

            (0, 13),  # LHWrist -> LRing1Knuckles
            (13, 14),  # LRing1Knuckles -> LRing2PIP
            (14, 15),  # LRing2PIP -> LRing3DIP
            (15, 16),  # LRing3DIP -> LRing4FingerTip

            (0, 17),  # LHWrist -> LPinky1Knuckles
            (17, 18),  # LPinky1Knuckles -> LPinky2PIP
            (18, 19),  # LPinky2PIP -> LPinky3DIP
            (19, 20),  # LPinky3DIP -> LPinky4FingerTip
        ]

    @staticmethod
    def connections():
        return [
            # ------------- TORSO -------------
            (0, 1),  # RShoulder -> RElbow
            (1, 2),  # RElbow -> RWrist

            (3, 4),  # LShoulder -> LElbow
            (4, 5),  # LElbow -> LWrist

            (0, 3),  # RShoulder -> LShoulder
            (0, 6),  # RShoulder -> RHip
            (3, 7),  # LShoulder -> LHip
            (6, 7),  # RHip -> LHip

            # ------------- RH -------------
            (2, 8),  # RWrist
            (8, 9),  # RHWrist -> RThumb1CMC
            (9, 10),  # RThumb1CMC -> RThumb2Knuckles
            (10, 11),  # RThumb2Knuckles -> RThumb3IP
            (11, 12),  # RThumb3IP -> RThumb4FingerTip

            (8, 13),  # RHWrist -> RIndex1Knuckles
            (13, 14),  # RIndex1Knuckles -> RIndex2PIP
            (14, 15),  # RIndex2PIP -> RIndex3DIP
            (15, 16),  # RIndex3DIP -> RIndex4FingerTip

            (8, 17),  # RHWrist -> RMiddle1Knuckles
            (17, 18),  # RMiddle1Knuckles -> RMiddle2PIP
            (18, 19),  # RMiddle2PIP -> RMiddle3DIP
            (19, 20),  # RMiddle3DIP -> RMiddle4FingerTip

            (8, 21),  # RHWrist -> RRing1Knuckles
            (21, 22),  # RRing1Knuckles -> RRing2PIP
            (22, 23),  # RRing2PIP -> RRing3DIP
            (23, 24),  # RRing3DIP -> RRing4FingerTip

            (8, 25),  # RHWrist -> RPinky1Knuckles
            (25, 26),  # RPinky1Knuckles -> RPinky2PIP
            (26, 27),  # RPinky2PIP -> RPinky3DIP
            (27, 28),  # RPinky3DIP -> RPinky4FingerTip

            # ------------- LH -------------
            (5, 29),  # LWrist
            (29, 30),  # LHWrist -> LThumb1CMC
            (30, 31),  # LThumb1CMC -> LThumb2Knuckles
            (31, 32),  # LThumb2Knuckles -> LThumb3IP
            (32, 33),  # LThumb3IP -> LThumb4FingerTip

            (29, 34),  # LHWrist -> LIndex1Knuckles
            (34, 35),  # LIndex1Knuckles -> LIndex2PIP
            (35, 36),  # LIndex2PIP -> LIndex3DIP
            (36, 37),  # LIndex3DIP -> LIndex4FingerTip

            (29, 38),  # LHWrist -> LMiddle1Knuckles
            (38, 39),  # LMiddle1Knuckles -> LMiddle2PIP
            (39, 40),  # LMiddle2PIP -> LMiddle3DIP
            (40, 41),  # LMiddle3DIP -> LMiddle4FingerTip

            (29, 42),  # LHWrist -> LRing1Knuckles
            (42, 43),  # LRing1Knuckles -> LRing2PIP
            (43, 44),  # LRing2PIP -> LRing3DIP
            (44, 45),  # LRing3DIP -> LRing4FingerTip

            (29, 46),  # LHWrist -> LPinky1Knuckles
            (46, 47),  # LPinky1Knuckles -> LPinky2PIP
            (47, 48),  # LPinky2PIP -> LPinky3DIP
            (48, 49),  # LPinky3DIP -> LPinky4FingerTip
        ]

    @staticmethod
    def ids_torsoarms():
        # torso + RH wrist + LH wrist
        return list(range(0, 8)) + [8] + [29]

    @staticmethod
    def ids_rh():
        return list(range(8, 29))

    @staticmethod
    def ids_lh():
        return list(range(29, 50))

    @staticmethod
    def nodes_dict():
        return {
            # ------------- TORSO -------------
            'RShoulder': 0,
            'RElbow': 1,
            'RWrist': 2,
            'LShoulder': 3,
            'LElbow': 4,
            'LWrist': 5,
            'RHip': 6,
            'LHip': 7,

            # ------------- RH -------------
            'RHWrist': 8,
            'RThumb1CMC': 9,
            'RThumb2Knuckles': 10,
            'RThumb3IP': 11,
            'RThumb4FingerTip': 12,
            'RIndex1Knuckles': 13,
            'RIndex2PIP': 14,
            'RIndex3DIP': 15,
            'RIndex4FingerTip': 16,
            'RMiddle1Knuckles': 17,
            'RMiddle2PIP': 18,
            'RMiddle3DIP': 19,
            'RMiddle4FingerTip': 20,
            'RRing1Knuckles': 21,
            'RRing2PIP': 22,
            'RRing3DIP': 23,
            'RRing4FingerTip': 24,
            'RPinky1Knuckles': 25,
            'RPinky2PIP': 26,
            'RPinky3DIP': 27,
            'RPinky4FingerTip': 28,

            # ------------- LH -------------
            'LHWrist': 29,
            'LThumb1CMC': 30,
            'LThumb2Knuckles': 31,
            'LThumb3IP': 32,
            'LThumb4FingerTip': 33,
            'LIndex1Knuckles': 34,
            'LIndex2PIP': 35,
            'LIndex3DIP': 36,
            'LIndex4FingerTip': 37,
            'LMiddle1Knuckles': 38,
            'LMiddle2PIP': 39,
            'LMiddle3DIP': 40,
            'LMiddle4FingerTip': 41,
            'LRing1Knuckles': 42,
            'LRing2PIP': 43,
            'LRing3DIP': 44,
            'LRing4FingerTip': 45,
            'LPinky1Knuckles': 46,
            'LPinky2PIP': 47,
            'LPinky3DIP': 48,
            'LPinky4FingerTip': 49,
        }
