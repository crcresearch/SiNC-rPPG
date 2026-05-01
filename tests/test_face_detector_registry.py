from __future__ import annotations

import pytest

from preprocessing.face_detector import get_face_detector


def test_mediapipe_detector_registered() -> None:
    d = get_face_detector("mediapipe")
    assert d.name == "mediapipe"


def test_unknown_detector_raises() -> None:
    with pytest.raises(ValueError, match="Unknown face detector"):
        get_face_detector("no_such_backend")


def test_retinaface_not_implemented() -> None:
    d = get_face_detector("retinaface")
    with pytest.raises(NotImplementedError):
        d.landmark_directory(".")
