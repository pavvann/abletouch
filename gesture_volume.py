"""
AbleTouch – control Ableton track volumes with hand gestures via webcam.

Left hand  → CC 7 on MIDI channel 1 (Track 1)
Right hand → CC 7 on MIDI channel 2 (Track 2)

Pinch distance (thumb tip ↔ index tip), normalised by hand size,
maps to MIDI CC 0-127 with exponential smoothing.
"""

import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import rtmidi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_PINCH_RATIO = 0.10
MAX_PINCH_RATIO = 0.60
SMOOTH_ALPHA    = 0.30
DEADBAND        = 2
CC_NUMBER       = 7

LEFT_CC_CHANNEL  = 0
RIGHT_CC_CHANNEL = 1

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

BAR_WIDTH  = 30
BAR_HEIGHT = 200
BAR_Y_TOP  = 60
LEFT_BAR_X = 20

FONT        = cv2.FONT_HERSHEY_SIMPLEX
COLOR_LEFT  = (0,   220, 100)
COLOR_RIGHT = (0,   140, 255)
COLOR_LINE  = (255, 255,   0)
COLOR_FPS   = (200, 200, 200)
COLOR_INFO  = (180, 180, 180)


def ensure_model() -> str:
    if not MODEL_PATH.exists():
        print(f"[INFO] Downloading model → {MODEL_PATH} …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Done.")
    return str(MODEL_PATH)


def open_midi_port() -> rtmidi.MidiOut:
    midiout = rtmidi.MidiOut()
    ports   = midiout.get_ports()
    iac_idx = next((i for i, n in enumerate(ports) if "IAC" in n), None)
    if iac_idx is not None:
        midiout.open_port(iac_idx)
        print(f"[MIDI] Opened: {ports[iac_idx]}")
    else:
        midiout.open_virtual_port("AbleTouch")
        print("[MIDI] Virtual port 'AbleTouch'")
    return midiout


def send_cc(midiout, channel: int, cc: int, value: int) -> None:
    midiout.send_message([0xB0 | (channel & 0x0F), cc & 0x7F, value & 0x7F])


def lm_px(lm, w: int, h: int) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def euclidean(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_normalised_pinch(landmarks, w: int, h: int) -> float:
    thumb_tip = lm_px(landmarks[4], w, h)
    index_tip = lm_px(landmarks[8], w, h)
    wrist     = lm_px(landmarks[0], w, h)
    mid_mcp   = lm_px(landmarks[9], w, h)
    scale = euclidean(wrist, mid_mcp)
    if scale < 1e-6:
        return 0.0
    return euclidean(thumb_tip, index_tip) / scale


def normalised_pinch_to_cc(ratio: float) -> int:
    t = (ratio - MIN_PINCH_RATIO) / (MAX_PINCH_RATIO - MIN_PINCH_RATIO)
    return int(round(max(0.0, min(1.0, t)) * 127))


def draw_skeleton(frame, landmarks, w, h, color):
    pts = [lm_px(lm, w, h) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (160, 160, 160), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 3, color, -1)


def draw_pinch_line(frame, landmarks, w, h, cc_value, color):
    thumb_tip = lm_px(landmarks[4], w, h)
    index_tip = lm_px(landmarks[8], w, h)
    cv2.line(frame, thumb_tip, index_tip, COLOR_LINE, 2, cv2.LINE_AA)
    cv2.circle(frame, thumb_tip, 6, COLOR_LINE, -1)
    cv2.circle(frame, index_tip, 6, COLOR_LINE, -1)
    mid_x = (thumb_tip[0] + index_tip[0]) // 2
    mid_y = (thumb_tip[1] + index_tip[1]) // 2 - 15
    cv2.putText(frame, f"{int(cc_value/127*100)}%", (mid_x - 18, mid_y),
                FONT, 0.65, color, 2, cv2.LINE_AA)


def draw_volume_bar(frame, cc_value, x, label, color):
    fill_h = int((cc_value / 127.0) * BAR_HEIGHT)
    x1, x2 = x, x + BAR_WIDTH
    y_bot   = BAR_Y_TOP + BAR_HEIGHT
    y_fill  = y_bot - fill_h
    cv2.rectangle(frame, (x1, BAR_Y_TOP), (x2, y_bot), (50, 50, 50), -1)
    if fill_h > 0:
        cv2.rectangle(frame, (x1, y_fill), (x2, y_bot), color, -1)
    cv2.rectangle(frame, (x1, BAR_Y_TOP), (x2, y_bot), color, 1)
    cv2.putText(frame, label, (x1, BAR_Y_TOP - 22), FONT, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(cc_value/127*100)}%", (x1, y_bot + 18),
                FONT, 0.45, color, 1, cv2.LINE_AA)


def main():
    model_path = ensure_model()
    midiout    = open_midi_port()

    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options   = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smooth    = {"Left": 64.0, "Right": 64.0}
    last_sent = {"Left": -1,   "Right": -1}

    t0        = time.time()
    prev_time = t0
    print("[INFO] AbleTouch running. Pinch to control volume. ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms   = int((time.time() - t0) * 1000)
        results = detector.detect_for_video(mp_img, ts_ms)

        detected: set[str] = set()

        if results.hand_landmarks and results.handedness:
            for hand_lms, handedness in zip(results.hand_landmarks, results.handedness):
                label = handedness[0].category_name
                detected.add(label)

                ratio  = compute_normalised_pinch(hand_lms, w, h)
                cc_raw = normalised_pinch_to_cc(ratio)

                smooth[label] = SMOOTH_ALPHA * cc_raw + (1.0 - SMOOTH_ALPHA) * smooth[label]
                cc_val = int(round(smooth[label]))

                if abs(cc_val - last_sent[label]) > DEADBAND:
                    ch = LEFT_CC_CHANNEL if label == "Left" else RIGHT_CC_CHANNEL
                    send_cc(midiout, ch, CC_NUMBER, cc_val)
                    last_sent[label] = cc_val

                color = COLOR_LEFT if label == "Left" else COLOR_RIGHT
                draw_skeleton(frame, hand_lms, w, h, color)
                draw_pinch_line(frame, hand_lms, w, h, cc_val, color)

        right_bar_x = w - LEFT_BAR_X - BAR_WIDTH
        left_cc     = last_sent["Left"]  if last_sent["Left"]  >= 0 else int(smooth["Left"])
        right_cc    = last_sent["Right"] if last_sent["Right"] >= 0 else int(smooth["Right"])
        left_color  = COLOR_LEFT  if "Left"  in detected else (0, 80, 40)
        right_color = COLOR_RIGHT if "Right" in detected else (0, 55, 100)
        draw_volume_bar(frame, left_cc,  LEFT_BAR_X,  "TRACK 1", left_color)
        draw_volume_bar(frame, right_cc, right_bar_x, "TRACK 2", right_color)

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 40), FONT, 0.5, COLOR_FPS, 1, cv2.LINE_AA)
        cv2.putText(frame, "ESC to quit  |  pinch to control volume",
                    (w // 2 - 160, h - 15), FONT, 0.5, COLOR_INFO, 1, cv2.LINE_AA)

        cv2.imshow("AbleTouch", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    midiout.close_port()
    del midiout


if __name__ == "__main__":
    main()
