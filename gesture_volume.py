"""
AbleTouch – control Ableton track volumes with hand gestures via webcam.

Left hand:   index→T1, middle→T2, ring→T3, pinky→T4  (CC 7, ch 1-4)
Right hand:  index→T5, middle→T6, ring→T7, pinky→T8  (CC 7, ch 5-8)

Mode 1 – Curl:   curl each finger down to raise its track volume
Mode 2 – Height: touch thumb to a finger to select/deselect that track,
                 then move your whole hand up/down to control selected tracks
"""

import math
import time
import threading
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import rtmidi
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
SMOOTH_ALPHA = 0.50
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

# Mode 1 tuning
MIN_CURL = 0.05
MAX_CURL = 0.65

# Mode 2 tuning
TOUCH_THRESHOLD = 0.28   # normalised thumb-finger distance to register a touch
HEIGHT_MIN      = 0.20   # wrist Y (normalised, 0=top) that maps to CC 127 (hand high)
HEIGHT_MAX      = 0.80   # wrist Y that maps to CC 0   (hand low)

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

DETECT_W       = 640
DETECT_H       = 360
FIRE_THRESHOLD = 15
EMOJI_SIZE     = 40


# ---------------------------------------------------------------------------
# Emoji
# ---------------------------------------------------------------------------

def build_emoji_img(emoji: str, size: int) -> np.ndarray:
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", size)
    except OSError:
        return np.zeros((size, size, 4), dtype=np.uint8)
    canvas = Image.new("RGBA", (size + 8, size + 8), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).text((4, 4), emoji, font=font, embedded_color=True)
    return np.array(canvas)


def overlay_emoji(frame, emoji_img, cx, cy):
    eh, ew = emoji_img.shape[:2]
    x1, y1 = cx - ew // 2, cy - eh // 2
    x2, y2 = x1 + ew, y1 + eh
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if fx2 <= fx1 or fy2 <= fy1:
        return
    ex1, ey1 = fx1 - x1, fy1 - y1
    ex2, ey2 = ex1 + (fx2 - fx1), ey1 + (fy2 - fy1)
    alpha = emoji_img[ey1:ey2, ex1:ex2, 3:4] / 255.0
    rgb   = emoji_img[ey1:ey2, ex1:ex2, :3][:, :, ::-1]
    roi   = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    frame[fy1:fy2, fx1:fx2] = (roi * (1 - alpha) + rgb * alpha).astype(np.uint8)


# ---------------------------------------------------------------------------
# Camera reader
# ---------------------------------------------------------------------------

class CameraReader:
    def __init__(self, cap):
        self._cap   = cap
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            return self._frame

    def release(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def lm_3d(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def hand_scale(landmarks) -> float:
    wrist   = lm_3d(landmarks[0])
    mid_mcp = lm_3d(landmarks[9])
    return float(np.linalg.norm(mid_mcp - wrist)) or 1e-6


# Mode 1: PIP curl angle
def pip_curl(landmarks, tip_i, dip_i, pip_i, mcp_i) -> float:
    mcp = lm_3d(landmarks[mcp_i])
    pip = lm_3d(landmarks[pip_i])
    tip = lm_3d(landmarks[tip_i])
    v1, v2 = pip - mcp, tip - pip
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.clip((1.0 - np.dot(v1/n1, v2/n2)) / 2.0, 0.0, 1.0))


def curl_to_cc(curl: float) -> int:
    t = (curl - MIN_CURL) / (MAX_CURL - MIN_CURL)
    return int(round(np.clip(t, 0.0, 1.0) * 127))


# Mode 2: thumb-finger touch + wrist height
def thumb_touching(landmarks, finger_tip_i: int) -> bool:
    thumb = lm_3d(landmarks[4])
    tip   = lm_3d(landmarks[finger_tip_i])
    dist  = float(np.linalg.norm(thumb - tip))
    return dist / hand_scale(landmarks) < TOUCH_THRESHOLD


def wrist_height_cc(landmarks) -> int:
    """Hand raised = CC 127, hand lowered = CC 0."""
    y = landmarks[0].y   # 0=top of frame, 1=bottom
    t = (y - HEIGHT_MIN) / (HEIGHT_MAX - HEIGHT_MIN)
    return int(round(np.clip(1.0 - t, 0.0, 1.0) * 127))


# ---------------------------------------------------------------------------
# Drawing – shared
# ---------------------------------------------------------------------------

def draw_skeleton(frame, landmarks, w, h):
    pts = [lm_px(lm, w, h) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (130, 130, 130), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 3, (200, 200, 200), -1)


def draw_bars(frame, cc_vals, side, active, w, highlight: set = None):
    """highlight: set of finger indices to draw at full brightness (mode 2 selected tracks)."""
    if highlight is None:
        highlight = {0, 1, 2, 3}
    total_w = 4 * BAR_W + 3 * BAR_GAP
    x0 = SIDE_PAD if side == "left" else w - SIDE_PAD - total_w
    for fi in range(4):
        x1, x2 = x0 + fi * (BAR_W + BAR_GAP), x0 + fi * (BAR_W + BAR_GAP) + BAR_W
        y_bot   = BAR_Y + BAR_H
        fill_h  = int((cc_vals[fi] / 127.0) * BAR_H)
        y_fill  = y_bot - fill_h
        base    = FINGER_COLORS[fi]
        if not active:
            color = tuple(c // 5 for c in base)
        elif fi not in highlight:
            color = tuple(c // 4 for c in base)   # dim unselected
        else:
            color = base
        cv2.rectangle(frame, (x1, BAR_Y), (x2, y_bot), (35, 35, 35), -1)
        if fill_h > 0:
            cv2.rectangle(frame, (x1, y_fill), (x2, y_bot), color, -1)
        cv2.rectangle(frame, (x1, BAR_Y), (x2, y_bot), color, 1)
        track_n = fi + 1 if side == "left" else fi + 5
        cv2.putText(frame, f"T{track_n}", (x1, BAR_Y - 6), FONT, 0.35, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{int(cc_vals[fi]/127*100)}%", (x1, y_bot + 14),
                    FONT, 0.35, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Drawing – mode 1
# ---------------------------------------------------------------------------

def draw_fingertips_m1(frame, landmarks, w, h, cc_vals, fire_img):
    for fi, (name, tip_i, dip_i, pip_i, mcp_i) in enumerate(FINGERS):
        pt    = lm_px(landmarks[tip_i], w, h)
        color = FINGER_COLORS[fi]
        pct   = int(cc_vals[fi] / 127 * 100)
        if cc_vals[fi] > FIRE_THRESHOLD:
            overlay_emoji(frame, fire_img, pt[0], pt[1])
        else:
            cv2.circle(frame, pt, 7, color, -1)
        cv2.putText(frame, f"{pct}%", (pt[0] - 14, pt[1] - EMOJI_SIZE // 2 - 4),
                    FONT, 0.42, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Drawing – mode 2
# ---------------------------------------------------------------------------

def draw_fingertips_m2(frame, landmarks, w, h, selected: set, cc_vals, fire_img):
    thumb_pt = lm_px(landmarks[4], w, h)
    for fi, (name, tip_i, dip_i, pip_i, mcp_i) in enumerate(FINGERS):
        pt    = lm_px(landmarks[tip_i], w, h)
        color = FINGER_COLORS[fi]
        pct   = int(cc_vals[fi] / 127 * 100)
        if fi in selected:
            overlay_emoji(frame, fire_img, pt[0], pt[1])
            # line from thumb to selected fingertip
            cv2.line(frame, thumb_pt, pt, color, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, pt, 7, tuple(c // 3 for c in color), -1)
        cv2.putText(frame, f"{pct}%", (pt[0] - 14, pt[1] - EMOJI_SIZE // 2 - 4),
                    FONT, 0.42, color if fi in selected else tuple(c//3 for c in color),
                    1, cv2.LINE_AA)

    # thumb dot
    cv2.circle(frame, thumb_pt, 7, (255, 255, 255), -1)


def draw_height_bar(frame, cc_val, side, active, w, h):
    """Vertical bar on the far edge showing current hand height."""
    bw    = 10
    bh    = BAR_H
    by    = BAR_Y + BAR_H + 30   # below the finger bars
    bx    = SIDE_PAD if side == "left" else w - SIDE_PAD - bw
    y_bot = by + bh
    fill  = int((cc_val / 127.0) * bh)

    color = (220, 220, 220) if active else (60, 60, 60)
    cv2.rectangle(frame, (bx, by), (bx + bw, y_bot), (35, 35, 35), -1)
    if fill > 0:
        cv2.rectangle(frame, (bx, y_bot - fill), (bx + bw, y_bot), color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, y_bot), color, 1)
    cv2.putText(frame, "H", (bx, by - 6), FONT, 0.35, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n  AbleTouch")
    print("  ---------")
    print("  1 – Curl mode   (curl each finger to control its track)")
    print("  2 – Height mode (touch thumb to finger to select, move hand up/down for volume)")
    print()
    while True:
        choice = input("  Select mode [1/2]: ").strip()
        if choice in ("1", "2"):
            mode = int(choice)
            break
        print("  Please enter 1 or 2.")
    print()

    model_path = ensure_model()
    midiout    = open_midi_port()
    fire_img   = build_emoji_img("🔥", EMOJI_SIZE)

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
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FPS,          60)
    reader = CameraReader(cap)

    # Shared state
    last_sent  = {"Left": [-1]*4, "Right": [-1]*4}
    ch_offset  = {"Left": 0,      "Right": 4}

    # Mode 1 state
    smooth_m1  = {"Left": [0.0]*4, "Right": [0.0]*4}

    # Mode 2 state
    smooth_h   = {"Left": 64.0,    "Right": 64.0}   # smoothed hand height CC
    selected   = {"Left": set(),   "Right": set()}   # which fingers are toggled on
    was_touch  = {"Left": [False]*4, "Right": [False]*4}
    cc_held    = {"Left": [0]*4,   "Right": [0]*4}   # last CC for each track in m2

    cv2.namedWindow(" ", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(" ", cv2.WND_PROP_TOPMOST, 1)

    t0 = prev_time = time.time()
    hint = ("curl fingers to control volume" if mode == 1
            else "touch thumb+finger to select track, move hand up/down for volume")
    print(f"[INFO] Mode {mode} active — {hint}. ESC to quit.")

    while True:
        frame = reader.read()
        if frame is None:
            time.sleep(0.001)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        small  = cv2.resize(frame, (DETECT_W, DETECT_H))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int((time.time() - t0) * 1000)
        results = detector.detect_for_video(mp_img, ts_ms)

        detected: set[str] = set()

        if results.hand_landmarks and results.handedness:
            for hand_lms, handedness in zip(results.hand_landmarks, results.handedness):
                label = handedness[0].category_name
                detected.add(label)
                draw_skeleton(frame, hand_lms, w, h)

                if mode == 1:
                    cc_vals = []
                    for fi, (name, tip_i, dip_i, pip_i, mcp_i) in enumerate(FINGERS):
                        curl   = pip_curl(hand_lms, tip_i, dip_i, pip_i, mcp_i)
                        cc_raw = curl_to_cc(curl)
                        smooth_m1[label][fi] = (SMOOTH_ALPHA * cc_raw
                                                + (1 - SMOOTH_ALPHA) * smooth_m1[label][fi])
                        cc_val = int(round(smooth_m1[label][fi]))
                        cc_vals.append(cc_val)
                        if abs(cc_val - last_sent[label][fi]) > DEADBAND:
                            send_cc(midiout, ch_offset[label] + fi, CC_NUMBER, cc_val)
                            last_sent[label][fi] = cc_val
                    draw_fingertips_m1(frame, hand_lms, w, h, cc_vals, fire_img)

                else:  # mode 2
                    # Detect thumb-finger touches and toggle selection
                    for fi, (name, tip_i, dip_i, pip_i, mcp_i) in enumerate(FINGERS):
                        touching = thumb_touching(hand_lms, tip_i)
                        if touching and not was_touch[label][fi]:   # rising edge → toggle
                            if fi in selected[label]:
                                selected[label].discard(fi)
                            else:
                                selected[label].add(fi)
                        was_touch[label][fi] = touching

                    # Smooth hand height
                    h_raw = wrist_height_cc(hand_lms)
                    smooth_h[label] = (SMOOTH_ALPHA * h_raw
                                       + (1 - SMOOTH_ALPHA) * smooth_h[label])
                    h_cc = int(round(smooth_h[label]))

                    # Apply height to selected tracks, hold others
                    for fi in range(4):
                        if fi in selected[label]:
                            cc_held[label][fi] = h_cc
                        cc_val = cc_held[label][fi]
                        if abs(cc_val - last_sent[label][fi]) > DEADBAND:
                            send_cc(midiout, ch_offset[label] + fi, CC_NUMBER, cc_val)
                            last_sent[label][fi] = cc_val

                    draw_fingertips_m2(frame, hand_lms, w, h,
                                       selected[label], cc_held[label], fire_img)
                    side = "left" if label == "Left" else "right"
                    draw_height_bar(frame, h_cc, side, True, w, h)

        # Volume bars
        left_cc  = [last_sent["Left"][i]  if last_sent["Left"][i]  >= 0 else 0 for i in range(4)]
        right_cc = [last_sent["Right"][i] if last_sent["Right"][i] >= 0 else 0 for i in range(4)]

        left_hi  = selected["Left"]  if mode == 2 else {0,1,2,3}
        right_hi = selected["Right"] if mode == 2 else {0,1,2,3}

        draw_bars(frame, left_cc,  "left",  "Left"  in detected, w, left_hi)
        draw_bars(frame, right_cc, "right", "Right" in detected, w, right_hi)

        # HUD
        now = time.time()
        cv2.putText(frame, f"FPS: {1/(now-prev_time+1e-9):.1f}", (10, h-35),
                    FONT, 0.5, COLOR_FPS, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Mode {mode}  |  ESC to quit  |  {hint}",
                    (w//2 - 260, h-12), FONT, 0.43, COLOR_INFO, 1, cv2.LINE_AA)
        prev_time = now

        cv2.imshow(" ", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    detector.close()
    reader.release()
    cap.release()
    cv2.destroyWindow(" ")
    midiout.close_port()
    del midiout


if __name__ == "__main__":
    main()
