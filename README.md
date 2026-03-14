# AbleTouch

touch your ableton

![demo placeholder](https://placehold.co/800x400/111/333?text=abletouch)

## How it works

Hold your palm toward the camera. Each finger controls one track — curl it down to raise the volume, straighten it to lower it.

```
Left hand   →   Tracks 1–4
Right hand  →   Tracks 5–8
```

Active fingers show a 🔥 instead of a dot.

## Setup

**1. Install dependencies**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Enable IAC Driver** (macOS virtual MIDI cable)

Audio MIDI Setup → Window → Show MIDI Studio → double-click IAC Driver → check **Device is online**

**3. Configure Ableton**

Preferences → Link/Tempo/MIDI → set IAC Driver **Track** and **Remote** to On

Then hit `⌘M` to enter MIDI map mode, click a track's volume knob, and curl the corresponding finger in front of the camera. Repeat for each track. Hit `⌘M` again to exit.

**4. Run**
```bash
python gesture_volume.py
```

Press `ESC` to quit.

## Tuning

All knobs are constants at the top of `gesture_volume.py`:

| Constant | Default | What it does |
|---|---|---|
| `MIN_CURL` | `0.05` | finger angle that = 0% volume |
| `MAX_CURL` | `0.65` | finger angle that = 100% volume (lower = less movement needed) |
| `SMOOTH_ALPHA` | `0.50` | responsiveness (higher = snappier, lower = smoother) |
| `FIRE_THRESHOLD` | `15` | CC value above which 🔥 appears |
