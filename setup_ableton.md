# AbleTouch – Setup Guide

Control Ableton Live track volumes with hand gestures via your webcam.

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended.

---

## 2. Set up a virtual MIDI port (macOS – IAC Driver)

AbleTouch sends MIDI CC messages, so you need a virtual MIDI cable between it and Ableton.

1. Open **Audio MIDI Setup** (Applications → Utilities → Audio MIDI Setup).
2. Choose **Window → Show MIDI Studio** (or press ⌘2).
3. Double-click **IAC Driver**.
4. Check **Device is online**.
5. Make sure at least one port (e.g. "Bus 1") exists. Add one with **+** if the list is empty.
6. Click **Apply** and close the window.

> On Windows you can use [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) to create a virtual port instead. AbleTouch will automatically detect and use any port whose name contains "IAC"; otherwise it creates a virtual port called **AbleTouch**.

---

## 3. Configure Ableton Live

### Enable the MIDI port

1. Open Ableton Live.
2. Go to **Live → Preferences → Link / Tempo / MIDI** (macOS) or **Options → Preferences → MIDI** (Windows).
3. In the **MIDI Ports** table, find **IAC Driver (Bus 1)** (or **AbleTouch** if using the virtual port).
4. Set both **Track** and **Remote** to **On** for that port.

### MIDI-map Track 1 volume to CC 7, Channel 1

1. Click the **MIDI** button in the top-right of Live's toolbar (or press **⌘M** / **Ctrl+M**) to enter MIDI Map mode. Controls that can be mapped turn blue.
2. Click the **volume knob** of Track 1 in the mixer.
3. Move your **left hand** in front of the webcam (run AbleTouch first) so that a CC 7 / Ch 1 message is received, or use Live's **MIDI From** selector and manually enter Channel 1, CC 7.
4. The mapping appears in the MIDI Map panel on the left.

### MIDI-map Track 2 volume to CC 7, Channel 2

Repeat the steps above for Track 2's volume knob, but this time move your **right hand** (or manually set Channel 2, CC 7).

5. Press **⌘M** / **Ctrl+M** again to leave MIDI Map mode.

> **Tip:** CC 7 is the standard "Channel Volume" controller. Some people prefer to map to a macro or a dedicated send knob instead — the process is identical.

---

## 4. Run AbleTouch

```bash
python gesture_volume.py
```

- A window opens showing your webcam feed.
- **Left hand** controls **Track 1** (green bar, left side).
- **Right hand** controls **Track 2** (orange bar, right side).
- **Pinch** your thumb and index finger together to lower the volume; spread them apart to raise it.
- The volume bars dim when a hand is not detected, and the last sent value is held (no stuck notes).
- Press **ESC** in the video window to quit.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Cannot open webcam" | Check that no other app is using the camera; try changing `cv2.VideoCapture(0)` to `1` or `2`. |
| No MIDI received in Ableton | Confirm IAC Driver is online and the port is enabled in Live's MIDI preferences. |
| Jittery volume | Move your hand more slowly, or lower `SMOOTH_ALPHA` in `gesture_volume.py` (e.g. `0.15`). |
| Volume doesn't reach 0 or 127 | Adjust `MIN_PINCH_RATIO` / `MAX_PINCH_RATIO` in `gesture_volume.py` to match your hand size. |
| Hand not detected | Improve lighting; keep your hand within ~0.5 – 1.5 m of the camera. |
