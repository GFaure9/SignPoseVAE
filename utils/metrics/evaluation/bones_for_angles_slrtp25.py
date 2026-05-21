RH_BONES_FOR_ANGLES_SLRTP25 = [
    (2, 8),  # RWrist
    (8, 9),  # RHWrist -> RThumb1CMC
    (9, 10),  # RThumb1CMC -> RThumb2Knuckles
    (10, 11),  # RThumb2Knuckles -> RThumb3IP
    (11, 12),  # RThumb3IP -> RThumb4FingerTip

    (8, 13),  # RHWrist -> RIndex1Knuckles
    (13, 14),  # RIndex1Knuckles -> RIndex2PIP
    (14, 15),  # RIndex2PIP -> RIndex3DIP
    (15, 16),  # RIndex3DIP -> RIndex4FingerTip
    
    (17, 18),  # RMiddle1Knuckles -> RMiddle2PIP
    (18, 19),  # RMiddle2PIP -> RMiddle3DIP
    (19, 20),  # RMiddle3DIP -> RMiddle4FingerTip

    (21, 22),  # RRing1Knuckles -> RRing2PIP
    (22, 23),  # RRing2PIP -> RRing3DIP
    (23, 24),  # RRing3DIP -> RRing4FingerTip

    (25, 26),  # RPinky1Knuckles -> RPinky2PIP
    (26, 27),  # RPinky2PIP -> RPinky3DIP
    (27, 28),  # RPinky3DIP -> RPinky4FingerTip
]

LH_BONES_FOR_ANGLES_SLRTP25 = [
    (5, 29),  # LWrist
    (29, 30),  # LWrist -> LThumb1CMC
    (30, 31),  # LThumb1CMC -> LThumb2Knuckles
    (31, 32),  # LThumb2Knuckles -> LThumb3IP
    (32, 33),  # LThumb3IP -> LThumb4FingerTip

    (29, 34),  # LWrist -> LIndex1Knuckles
    (34, 35),  # LIndex1Knuckles -> LIndex2PIP
    (35, 36),  # LIndex2PIP -> LIndex3DIP
    (36, 37),  # LIndex3DIP -> LIndex4FingerTip

    (38, 39),  # LMiddle1Knuckles -> LMiddle2PIP
    (39, 40),  # LMiddle2PIP -> LMiddle3DIP
    (40, 41),  # LMiddle3DIP -> LMiddle4FingerTip

    (42, 43),  # LRing1Knuckles -> LRing2PIP
    (43, 44),  # LRing2PIP -> LRing3DIP
    (44, 45),  # LRing3DIP -> LRing4FingerTip

    (46, 47),  # LPinky1Knuckles -> LPinky2PIP
    (47, 48),  # LPinky2PIP -> LPinky3DIP
    (48, 49),  # LPinky3DIP -> LPinky4FingerTip
]

HANDS_BONES_FOR_ANGLES_SLRTP25 = RH_BONES_FOR_ANGLES_SLRTP25 + LH_BONES_FOR_ANGLES_SLRTP25

BODY_BONES_FOR_ANGLES_SLRTP25 = [
    (0, 1),  # RShoulder -> RElbow
    (1, 2),  # RElbow -> RWrist

    (3, 4),  # LShoulder -> LElbow
    (4, 5),  # LElbow -> LWrist
]

BONES_FOR_ANGLES_SLRTP25 = [
    # ------------- TORSO -------------
    (0, 1),  # RShoulder -> RElbow
    (1, 2),  # RElbow -> RWrist

    (3, 4),  # LShoulder -> LElbow
    (4, 5),  # LElbow -> LWrist

    # === NOT COMPUTED ===
    # (0, 3),  # RShoulder -> LShoulder
    # (0, 6),  # RShoulder -> RHip
    # (3, 7),  # LShoulder -> LHip
    # (6, 7),  # RHip -> LHip
    # ====================

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
    
    # === NOT COMPUTED ===
    # (13, 17),  # RIndex1Knuckles -> RMiddle1Knuckles
    # ====================
    (17, 18),  # RMiddle1Knuckles -> RMiddle2PIP
    (18, 19),  # RMiddle2PIP -> RMiddle3DIP
    (19, 20),  # RMiddle3DIP -> RMiddle4FingerTip

    # === NOT COMPUTED ===
    # (17, 21),  # RMiddle1Knuckles -> RRing1Knuckles
    # ====================
    (21, 22),  # RRing1Knuckles -> RRing2PIP
    (22, 23),  # RRing2PIP -> RRing3DIP
    (23, 24),  # RRing3DIP -> RRing4FingerTip

    # === NOT COMPUTED ===
    # (21, 25),  # RRing1Knuckles -> RPinky1Knuckles
    # ====================
    (25, 26),  # RPinky1Knuckles -> RPinky2PIP
    (26, 27),  # RPinky2PIP -> RPinky3DIP
    (27, 28),  # RPinky3DIP -> RPinky4FingerTip

    # === NOT COMPUTED ===
    # (25, 8),  # RPinky1Knuckles -> RWrist
    # ====================

    # ------------- LH -------------
    (5, 29),  # LWrist
    (29, 30),  # LWrist -> LThumb1CMC
    (30, 31),  # LThumb1CMC -> LThumb2Knuckles
    (31, 32),  # LThumb2Knuckles -> LThumb3IP
    (32, 33),  # LThumb3IP -> LThumb4FingerTip

    (29, 34),  # LWrist -> LIndex1Knuckles
    (34, 35),  # LIndex1Knuckles -> LIndex2PIP
    (35, 36),  # LIndex2PIP -> LIndex3DIP
    (36, 37),  # LIndex3DIP -> LIndex4FingerTip

    # === NOT COMPUTED ===
    # (34, 38),  # LIndex1Knuckles -> LMiddle1Knuckles
    # ====================
    (38, 39),  # LMiddle1Knuckles -> LMiddle2PIP
    (39, 40),  # LMiddle2PIP -> LMiddle3DIP
    (40, 41),  # LMiddle3DIP -> LMiddle4FingerTip

    # === NOT COMPUTED ===
    # (38, 42),  # LMiddle1Knuckles -> LRing1Knuckles
    # ====================
    (42, 43),  # LRing1Knuckles -> LRing2PIP
    (43, 44),  # LRing2PIP -> LRing3DIP
    (44, 45),  # LRing3DIP -> LRing4FingerTip

    # === NOT COMPUTED ===
    # (42, 46),  # LRing1Knuckles -> LPinky1Knuckles
    # ====================
    (46, 47),  # LPinky1Knuckles -> LPinky2PIP
    (47, 48),  # LPinky2PIP -> LPinky3DIP
    (48, 49),  # LPinky3DIP -> LPinky4FingerTip

    # === NOT COMPUTED ===
    # (46, 29),  # LPinky1Knuckles -> LWrist
    # ====================
]