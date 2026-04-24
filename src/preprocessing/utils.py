import os

import cv2
import numpy as np

OUT_IMG_HEIGHT = 64
OUT_IMG_WIDTH = 64

try:
    from mesh_common import CANONICAL_LMRKS, face_mesh_to_array  # noqa: F401
except ImportError:
    from preprocessing.mesh_common import CANONICAL_LMRKS, face_mesh_to_array  # noqa: F401


def mediapipe_landmark_directory(frame_dir):
    """Backward-compatible entry point; delegates to the registered MediaPipe detector."""
    try:
        from face_detector import get_face_detector
    except ImportError:
        from preprocessing.face_detector import get_face_detector

    return get_face_detector("mediapipe").landmark_directory(frame_dir)


def mediapipe_landmark_video(video_path):
    """Backward-compatible entry point; delegates to the registered MediaPipe detector."""
    try:
        from face_detector import get_face_detector
    except ImportError:
        from preprocessing.face_detector import get_face_detector

    return get_face_detector("mediapipe").landmark_video(video_path)


def make_video_array_from_directory(
    vid_dir, lmrks, w=OUT_IMG_WIDTH, h=OUT_IMG_HEIGHT, dtype=np.uint8
):
    vid_len = len(lmrks)
    video_idx = 0
    frame_list = [os.path.join(vid_dir, f) for f in sorted(os.listdir(vid_dir))]
    output_video = np.zeros((vid_len, h, w, 3), dtype=dtype)
    successful = True

    for frame_path in frame_list:
        frame = cv2.imread(frame_path)
        if video_idx == 0:
            img_h, img_w = frame.shape[:2]

        if video_idx < vid_len:
            lmrk = lmrks[video_idx]
        else:  # lmrks are shorter than video
            successful = False
            print("ERROR: Fewer landmarks than video frames, must relandmark.")
            break

        lmrk = lmrk.astype(int)
        bbox = get_bbox(lmrk, img_w, img_h)
        square_bbox = get_square_bbox(bbox, img_w, img_h)

        x1, y1, x2, y2 = square_bbox
        cropped = frame[y1:y2, x1:x2]
        if cropped.size < 1:
            resized = np.zeros((h, w, 3), dtype=cropped.dtype)
        else:
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
        output_video[video_idx] = resized
        video_idx += 1

    if video_idx < vid_len:
        successful = False
        print(
            f"ERROR: Reached video idx {video_idx} while video was expected to be length {vid_len}."
        )

    return output_video, successful


def make_video_array(vid_path, lmrks, w=OUT_IMG_WIDTH, h=OUT_IMG_HEIGHT, dtype=np.uint8):
    vid_len = len(lmrks)
    cap = cv2.VideoCapture(vid_path, cv2.CAP_FFMPEG)
    video_idx = 0
    output_video = np.zeros((vid_len, h, w, 3), dtype=dtype)
    successful = True

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if video_idx == 0:
                img_h, img_w = frame.shape[:2]

            if video_idx < vid_len:
                lmrk = lmrks[video_idx]
            else:  # lmrks are shorter than video
                successful = False
                print("ERROR: Fewer landmarks than video frames, must relandmark.")
                break

            lmrk = lmrk.astype(int)
            bbox = get_bbox(lmrk, img_w, img_h)
            square_bbox = get_square_bbox(bbox, img_w, img_h)

            x1, y1, x2, y2 = square_bbox
            cropped = frame[y1:y2, x1:x2]
            if cropped.size < 1:
                resized = np.zeros((h, w, 3), dtype=cropped.dtype)
            else:
                resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
            output_video[video_idx] = resized
            video_idx += 1
        else:
            break

    cap.release()

    if video_idx < vid_len:
        successful = False
        print(
            f"ERROR: Reached video idx {video_idx} while video was expected to be length {vid_len}."
        )

    return output_video, successful


def get_bbox(lmrks, img_w, img_h):
    x_min, y_min = lmrks.min(axis=0)
    x_max, y_max = lmrks.max(axis=0)
    x_diff = x_max - x_min
    x_upper_pad = x_diff * 0.05
    x_lower_pad = x_diff * 0.05
    x_min -= x_upper_pad
    x_max += x_lower_pad
    if x_min < 0:
        x_min = 0
    if x_max > img_w:
        x_max = img_w
    y_diff = y_max - y_min
    y_upper_pad = y_diff * 0.3
    y_lower_pad = y_diff * 0.05
    y_min -= y_upper_pad
    y_max += y_lower_pad
    if y_min < 0:
        y_min = 0
    if y_max > img_h:
        y_max = img_h
    bbox = np.array([x_min, y_min, x_max, y_max]).astype(int)
    return bbox


def shift_inside_frame(x1, y1, x2, y2, img_w, img_h):
    if y1 < 0:
        y2 -= y1
        y1 -= y1
    if y2 > img_h:
        shift = y2 - img_h
        y1 -= shift
        y2 -= shift

    if x1 < 0:
        x2 -= x1
        x1 -= x1
    if x2 > img_w:
        shift = x2 - img_w
        x1 -= shift
        x2 -= shift

    return x1, y1, x2, y2


def get_square_bbox(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x1, y1, x2, y2 = shift_inside_frame(x1, y1, x2, y2, img_w, img_h)
    w = x2 - x1
    h = y2 - y1

    ## Push the rectangle out into a square
    if w > h:
        # if w > IN_IMG_HEIGHT:
        #     print('************** Oh no... **************')
        d = w - h
        pad = int(d / 2)
        y1 -= pad
        y2 += pad + (d % 2 == 1)
        x1, y1, x2, y2 = shift_inside_frame(x1, y1, x2, y2, img_w, img_h)
    elif w < h:
        # if h > IN_IMG_WIDTH:
        #     print('************** Oh no... **************')
        d = h - w
        pad = int(d / 2)
        x1 -= pad
        x2 += pad + (d % 2 == 1)
        x1, y1, x2, y2 = shift_inside_frame(x1, y1, x2, y2, img_w, img_h)

    if x1 < 0:
        x1 = 0
    if x2 > img_w:
        x2 = img_w
    if y1 < 0:
        y1 = 0
    if y2 > img_h:
        y2 = img_h

    w = x2 - x1
    h = y2 - y1
    return int(x1), int(y1), int(x2), int(y2)
