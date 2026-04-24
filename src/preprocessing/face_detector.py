"""Pluggable face landmark backends for preprocessing."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

try:
    from mesh_common import face_mesh_to_array
except ImportError:
    from preprocessing.mesh_common import face_mesh_to_array

_DETECTOR_REGISTRY: dict[str, type["FaceDetector"]] = {}


def register_detector(name: str, cls: type["FaceDetector"]) -> None:
    _DETECTOR_REGISTRY[name.lower()] = cls


def get_face_detector(name: str) -> "FaceDetector":
    key = (name or "mediapipe").lower().strip()
    if key not in _DETECTOR_REGISTRY:
        known = ", ".join(sorted(_DETECTOR_REGISTRY))
        raise ValueError(f"Unknown face detector {name!r}. Choose one of: {known}")
    return _DETECTOR_REGISTRY[key]()


class FaceDetector(ABC):
    """Abstract face landmark extractor for image directories or video files."""

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def landmark_directory(self, frame_dir: str | Path) -> np.ndarray:
        """Return ``(T, 68, 2)`` integer landmarks for sorted frames in ``frame_dir``."""

    def landmark_video(self, video_path: str | Path) -> np.ndarray:
        """Return ``(T, 68, 2)`` landmarks for each decoded frame of ``video_path``."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support video landmarking yet."
        )


class MediaPipeFaceDetector(FaceDetector):
    """MediaPipe FaceMesh (static images), matching the original preprocessing."""

    name = "mediapipe"

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        import mediapipe as mp

        self._mp = mp
        self._min_detection_confidence = min_detection_confidence

    def landmark_directory(self, frame_dir: str | Path) -> np.ndarray:
        frame_dir = Path(frame_dir)
        all_lmrks = []
        prev_lmrks = np.zeros((68, 2), dtype=np.int32)
        face_mesh = self._mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=self._min_detection_confidence,
        )
        frame_files = sorted(os.listdir(frame_dir))
        for frame_file in frame_files:
            frame_path = str(frame_dir / frame_file)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = frame.shape
            results = face_mesh.process(frame)
            lmrks = face_mesh_to_array(results, img_w, img_h)
            if lmrks is not None:
                prev_lmrks = lmrks
            else:
                lmrks = prev_lmrks
            all_lmrks.append(lmrks)
        return np.stack(all_lmrks)

    def landmark_video(self, video_path: str | Path) -> np.ndarray:
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        all_lmrks = []
        prev_lmrks = np.zeros((68, 2), dtype=np.int32)
        face_mesh = self._mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=self._min_detection_confidence,
        )
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = frame.shape
            results = face_mesh.process(frame)
            lmrks = face_mesh_to_array(results, img_w, img_h)
            if lmrks is not None:
                prev_lmrks = lmrks
            else:
                lmrks = prev_lmrks
            all_lmrks.append(lmrks)
        cap.release()
        return np.stack(all_lmrks)


class RetinaFaceDetector(FaceDetector):
    """Placeholder for RetinaFace-based landmarking or ROI boxes."""

    name = "retinaface"

    def landmark_directory(self, frame_dir: str | Path) -> np.ndarray:
        raise NotImplementedError(
            "RetinaFaceDetector is not implemented yet; use detector=mediapipe or add a backend."
        )


class MTCNNFaceDetector(FaceDetector):
    """Placeholder for MTCNN-based face detection."""

    name = "mtcnn"

    def landmark_directory(self, frame_dir: str | Path) -> np.ndarray:
        raise NotImplementedError(
            "MTCNNFaceDetector is not implemented yet; use detector=mediapipe or add a backend."
        )


register_detector("mediapipe", MediaPipeFaceDetector)
register_detector("retinaface", RetinaFaceDetector)
register_detector("mtcnn", MTCNNFaceDetector)
