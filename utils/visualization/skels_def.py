# ================= MP 178 =================
MP178_FACE_IDS = [i for i in range(50, 178)]
MP178_RH_IDS = [i for i in range(8, 29)]
MP178_LH_IDS = [i for i in range(29, 50)]

# format = (ID_Parent, ID_Child)
MP178_CONNECTIONS = [
    # ------------- TORSO -------------
    (0, 1, 0.36),  # RShoulder -> RElbow
    (1, 2, 0.27),  # RElbow -> RWrist

    (3, 4, 0.36),  # LShoulder -> LElbow
    (4, 5, 0.27),  # LElbow -> LWrist

    (0, 3, None),  # RShoulder -> LShoulder
    (0, 6, None),  # RShoulder -> RHip
    (3, 7, None),  # LShoulder -> LHip
    (6, 7, None),  # RHip -> LHip

    # ------------- RH -------------
    (2, 8, 0.01),  # RWrist
    (8, 9, 0.06),  # RHWrist -> RThumb1CMC
    (9, 10, 0.04),  # RThumb1CMC -> RThumb2Knuckles
    (10, 11, 0.03),  # RThumb2Knuckles -> RThumb3IP
    (11, 12, 0.025),  # RThumb3IP -> RThumb4FingerTip

    (8, 13, 0.12),  # RHWrist -> RIndex1Knuckles
    (13, 14, 0.05),  # RIndex1Knuckles -> RIndex2PIP
    (14, 15, 0.04),  # RIndex2PIP -> RIndex3DIP
    (15, 16, 0.025),  # RIndex3DIP -> RIndex4FingerTip

    (13, 17, 0.02),  # RIndex1Knuckles -> RMiddle1Knuckles
    (17, 18, 0.055),  # RMiddle1Knuckles -> RMiddle2PIP
    (18, 19, 0.04),  # RMiddle2PIP -> RMiddle3DIP
    (19, 20, 0.025),  # RMiddle3DIP -> RMiddle4FingerTip

    (17, 21, 0.02),  # RMiddle1Knuckles -> RRing1Knuckles
    (21, 22, 0.04),  # RRing1Knuckles -> RRing2PIP
    (22, 23, 0.035),  # RRing2PIP -> RRing3DIP
    (23, 24, 0.025),  # RRing3DIP -> RRing4FingerTip

    (21, 25, 0.02),  # RRing1Knuckles -> RPinky1Knuckles
    (25, 26, 0.035),  # RPinky1Knuckles -> RPinky2PIP
    (26, 27, 0.03),  # RPinky2PIP -> RPinky3DIP
    (27, 28, 0.025),  # RPinky3DIP -> RPinky4FingerTip

    (25, 8, None),  # RPinky1Knuckles -> RWrist

    # ------------- LH -------------
    (5, 29, 0.01),  # LWrist
    (29, 30, 0.06),  # LWrist -> LThumb1CMC
    (30, 31, 0.04),  # LThumb1CMC -> LThumb2Knuckles
    (31, 32, 0.03),  # LThumb2Knuckles -> LThumb3IP
    (32, 33, 0.025),  # LThumb3IP -> LThumb4FingerTip

    (29, 34, 0.12),  # LWrist -> LIndex1Knuckles
    (34, 35, 0.05),  # LIndex1Knuckles -> LIndex2PIP
    (35, 36, 0.04),  # LIndex2PIP -> LIndex3DIP
    (36, 37, 0.025),  # LIndex3DIP -> LIndex4FingerTip

    (34, 38, 0.02),  # LIndex1Knuckles -> LMiddle1Knuckles
    (38, 39, 0.055),  # LMiddle1Knuckles -> LMiddle2PIP
    (39, 40, 0.04),  # LMiddle2PIP -> LMiddle3DIP
    (40, 41, 0.025),  # LMiddle3DIP -> LMiddle4FingerTip

    (38, 42, 0.02),  # LMiddle1Knuckles -> LRing1Knuckles
    (42, 43, 0.04),  # LRing1Knuckles -> LRing2PIP
    (43, 44, 0.035),  # LRing2PIP -> LRing3DIP
    (44, 45, 0.025),  # LRing3DIP -> LRing4FingerTip

    (42, 46, 0.02),  # LRing1Knuckles -> LPinky1Knuckles
    (46, 47, 0.035),  # LPinky1Knuckles -> LPinky2PIP
    (47, 48, 0.03),  # LPinky2PIP -> LPinky3DIP
    (48, 49, 0.025),  # LPinky3DIP -> LPinky4FingerTip

    (46, 29, None),  # LPinky1Knuckles -> LWrist

    # ------------- FACE -------------
    (126, 127, None),
    (127, 96, None),
    (96, 97, None),
    (97, 150, None),
    (150, 151, None),
    (151, 169, None),
    (169, 114, None),
    (114, 115, None),
    (115, 116, None),
    (116, 171, None),
    (171, 133, None),
    (133, 88, None),
    (88, 89, None),
    (89, 93, None),
    (93, 94, None),
    (94, 67, None),
    (67, 68, None),
    (68, 132, None),
    (132, 117, None),
    (117, 118, None),
    (118, 177, None),
    (177, 145, None),
    (145, 134, None),
    (134, 135, None),
    (135, 137, None),
    (137, 75, None),
    (75, 76, None),
    (76, 139, None),
    (139, 98, None),
    (98, 99, None),
    (99, 136, None),
    (136, 109, None),
    (109, 110, None),
    (110, 154, None),
    (154, 174, None),
    (174, 126, None),
    (147, 77, None),
    (77, 78, None),
    (78, 152, None),
    (152, 153, None),
    (153, 122, None),
    (122, 121, None),
    (121, 144, None),
    (144, 143, None),
    (143, 161, None),
    (161, 147, None),
    (70, 69, None),
    (69, 146, None),
    (146, 57, None),
    (57, 56, None),
    (56, 158, None),
    (158, 111, None),
    (111, 112, None),
    (112, 155, None),
    (155, 176, None),
    (176, 70, None),
    (82, 81, None),
    (81, 125, None),
    (125, 124, None),
    (124, 106, None),
    (106, 105, None),
    (105, 164, None),
    (164, 173, None),
    (173, 141, None),
    (141, 142, None),
    (142, 128, None),
    (128, 129, None),
    (129, 168, None),
    (168, 113, None),
    (113, 79, None),
    (79, 80, None),
    (80, 87, None),
    (87, 86, None),
    (86, 163, None),
    (163, 162, None),
    (162, 82, None),
    (95, 130, None),
    (130, 55, None),
    (55, 54, None),
    (54, 84, None),
    (84, 51, None),
    (51, 50, None),
    (50, 62, None),
    (62, 61, None),
    (61, 167, None),
    (167, 166, None),
    (166, 172, None),
    (172, 175, None),
    (175, 140, None),
    (140, 58, None),
    (58, 59, None),
    (59, 123, None),
    (123, 165, None),
    (165, 91, None),
    (91, 92, None),
    (92, 95, None)
]

# ==========================================
