"""Shared MediaPipe FaceMesh landmark indexing (used by utils and face_detector)."""

from __future__ import annotations

import numpy as np

# Mapping from Mediapipe FaceMesh coordinates to OpenFace-style indices (68 points).
CANONICAL_LMRKS = [
    162,
    234,
    93,
    58,
    172,
    136,
    149,
    148,
    152,
    377,
    378,
    365,
    397,
    288,
    323,
    454,
    389,
    71,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    301,
    168,
    197,
    5,
    4,
    75,
    97,
    2,
    326,
    305,
    33,
    160,
    158,
    133,
    153,
    144,
    362,
    385,
    387,
    263,
    373,
    380,
    61,
    39,
    37,
    0,
    267,
    269,
    291,
    405,
    314,
    17,
    84,
    181,
    78,
    82,
    13,
    312,
    308,
    317,
    14,
    87,
]


def face_mesh_to_array(results, img_w, img_h):
    if not results.multi_face_landmarks:
        lmrks = None
    else:
        lmrks = np.array(
            [
                [
                    results.multi_face_landmarks[0].landmark[i].x,
                    results.multi_face_landmarks[0].landmark[i].y,
                ]
                for i in CANONICAL_LMRKS
            ]
        )
        lmrks = (lmrks * [img_w, img_h]).astype(int)
    return lmrks
