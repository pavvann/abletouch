"""
AbleTouch – palm facing camera, each finger you curl down controls one track.

Left hand:   index→T1, middle→T2, ring→T3, pinky→T4  (CC 7, ch 1-4)
Right hand:  index→T5, middle→T6, ring→T7, pinky→T8  (CC 7, ch 5-8)

Finger straight = 0%, finger curled down = 100%.
Uses PIP joint bend angle in 3D – orientation-independent, per-finger isolated.
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
# Tuning
# ---------------------------------------------------------------------------
MIN_CURL     = 0.05
MAX_CURL     = 0.65
SMOOTH_ALPHA = 0.30
DEADBAND     = 2
CC_NUMBER    = 7

FINGERS = [
    ("INDEX",  8,  7,  6,  5),
    ("MIDDLE", 12, 11, 10,  9),
    ("RING",   16, 15, 14, 13),
    ("PINKY",  20, 19, 18, 17),
]

FINGER_COLORS = [
    ( 80, 220, 255),
    ( 80, 255, 120),
    (255, 200,  60),
    (200,  80, 255),
]

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

FONT       = cv2.FONT_HERSHEY_SIMPLEX
COLOR_FPS  = (200, 200, 200)
COLOR_INFO = (180, 180, 180)

BAR_W    = 18
BAR_GAP  = 8
BAR_H    = 200
BAR_Y    = 50
SIDE_PAD = 16


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
        print(f"[MIDI] {ports[iac_idx]}")
    else:
        midiout.open_virtual_port("AbleTouch")
        print("[MIDI] Virtual port 'AbleTouch'")
    return midiout


def send_cc(midiout, ch, cc, val):
    midiout.send_message([0xB0 | (ch & 0x0F), cc & 0x7F, val & 0x7F])


def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def lm_3d(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def pip_curl(landmarks, tip_i, dip_i, pip_i, mcp_i) -> float:
    mcp = lm_3d(landmarks[mcp_i])
    pip = lm_3d(landmarks[pip_i])
    tip = lm_3d(landmarks[tip_i])
    v_prox = pip - mcp
    v_dist = tip - pip
    n1, n2 = np.linalg.norm(v_prox), np.linalg.norm(v_dist)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot  = float(np.dot(v_prox / n1, v_dist / n2))
    return float(np.clip((1.0 - dot) / 2.0, 0.0, 1.0))


def curl_to_cc(curl: float) -> int:
    t = (curl - MIN_CURL) / (MAX_CURL - MIN_CURL)
    return int(round(np.clip(t, 0.0, 1.0) * 127))


def draw_skeleton(frame, landmarks, w, h):
    pts = [lm_px(lm, w, h) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (130, 130, 130), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 3, (200, 200, 200), -1)


def draw_fingertips(frame, landmarks, w, h, cc_vals):
    for fi, (name, tip_i, dip_i, pip_i, mcp_i) in enumerate(FINGERS):
        pt     = lm_px(landmarks[tip_i], w, h)
        color  = FINGER_COLORS[fi]
        pct    = int(cc_vals[fi] / 127 * 100)
        radius = 6 + int(pct / 100 * 8)
        cv2.circle(frame, pt, radius, color, -1)
        cv2.putText(frame, f"{pct}%", (pt[0] - 14, pt[1] - radius - 4),
                    FONT, 0.42, color, 1, cv2.LINE_AA)


def draw_bars(frame, cc_vals, side, active, w):
    total_w = 4 * BAR_W + 3 * BAR_GAP
    x0 = SIDE_PAD if side == "left" else w - SIDE_PAD - total_w
    for fi in range(4):
        x1    = x0 + fi * (BAR_W + BAR_GAP)
        x2    = x1 + BAR_W
        y_bot = BAR_Y + BAR_H
        fill_h = int((cc_vals[fi] / 127.0) * BAR_H)
        y_fill = y_bot - fill_h
        base   = FINGER_COLORS[fi]
        color  = base if active else tuple(c // 5 for c in base)
        cv2.rectangle(frame, (x1, BAR_Y), (x2, y_bot), (35, 35, 35), -1)
        if fill_h > 0:
            cv2.rectangle(frame, (x1, y_fill), (x2, y_bot), color, -1)
        cv2.rectangle(frame, (x1, BAR_Y), (x2, y_bot), color, 1)
        track_n = fi + 1 if side == "left" else fi + 5
        cv2.putText(frame, f"T{track_n}", (x1, BAR_Y - 6), FONT, 0.35, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{int(cc_vals[fi]/127*100)}%", (x1, y_bot + 14),
                    FONT, 0.35, color, 1, cv2.LINE_AA)


def main():
    model_path = ensure_model()
    midiout    = open_midi_port()

    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options   = mp_vision.HandLandmarkerOptions(
        base_options=base_opts, num_hands=2,
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

    smooth    = {"Left": [0.0]*4, "Right": [0.0]*4}
    last_sent = {"Left": [-1]*4,  "Right": [-1]*4}
    ch_offset = {"Left": 0,       "Right": 4}

    t0 = prev_time = time.time()
    print("[INFO] Hold palm toward camera. Curl each finger to control its track. ESC to quit.")

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
                draw_skeleton(frame, hand_lms, w, h)
                cc_vals = []
                for fi, (name, tip_i, dip_i, pip_i, mcp_i) in enumerate(FINGERS):
                    curl   = pip_curl(hand_lms, tip_i, dip_i, pip_i, mcp_i)
                    cc_raw = curl_to_cc(curl)
                    smooth[label][fi] = SMOOTH_ALPHA * cc_raw + (1 - SMOOTH_ALPHA) * smooth[label][fi]
                    cc_val = int(round(smooth[label][fi]))
                    cc_vals.append(cc_val)
                    if abs(cc_val - last_sent[label][fi]) > DEADBAND:
                        send_cc(midiout, ch_offset[label] + fi, CC_NUMBER, cc_val)
                        last_sent[label][fi] = cc_val
                draw_fingertips(frame, hand_lms, w, h, cc_vals)

        left_vals  = [last_sent["Left"][i]  if last_sent["Left"][i]  >= 0 else int(smooth["Left"][i])  for i in range(4)]
        right_vals = [last_sent["Right"][i] if last_sent["Right"][i] >= 0 else int(smooth["Right"][i]) for i in range(4)]
        draw_bars(frame, left_vals,  "left",  "Left"  in detected, w)
        draw_bars(frame, right_vals, "right", "Right" in detected, w)

        now = time.time()
        cv2.putText(frame, f"FPS: {1/(now-prev_time+1e-9):.1f}", (10, h-35),
                    FONT, 0.5, COLOR_FPS, 1, cv2.LINE_AA)
        cv2.putText(frame, "ESC to quit  |  curl fingers to control volume",
                    (w//2 - 210, h-12), FONT, 0.45, COLOR_INFO, 1, cv2.LINE_AA)
        prev_time = now

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
