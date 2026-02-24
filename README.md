# WhereDaMilk
### AI-Powered Vision Assistant

<img src="/assets/icon.jpg" width="400" alt="WhereDaMilk">

Built to help visually impaired users navigate and understand their surroundings in real-world environments.

[▶ Watch Demo](http://www.youtube.com/watch?v=gTKes_XwjXY)

---

## What It Does

wheredamilk helps you locate and learn about objects in real time using your webcam and microphone. Speak a command, and the app scans the scene, identifies the target, tracks it, and narrates its position.

The app has **4 modes**, each triggered by voice:

| Mode | Command | What happens |
|---|---|---|
| **FIND** | `"find milk"` | Scans for the object, locks on, and tracks it silently. Announces location: *"Found milk on your left — keep going."* |
| **WHAT** | `"what is this"` | Identifies the object in frame, announces its class and position. |
| **READ** | `"read"` | OCRs the largest visible object and reads any text aloud. |
| **DETAILS** | `"tell me more"` | Sends the current frame to Gemini Vision for a full product analysis (brand, ingredients, info). |

**FIND mode** uses two-stage matching:
1. YOLO class matching (fast, for common objects)
2. OCR text fallback (for labeled products like "COCA-COLA", "JUICE")

Navigation guidance combines **MiDaS monocular depth estimation** with horizontal position to give spoken cues like *"on your left — almost there"* or *"stop, it's right in front of you"*.

---

## Architecture

```
Microphone
    │
    ▼
SpeechListener (utils/speech.py)
    │  voice command ("find milk")
    ▼
Mode Handler (logic/modes.py)
    │
    ├── FIND ──────────────────────────────────────────────────────────┐
    │   └── YOLOv8n (vision/yolo.py)                                   │
    │       ├── Class match → logic/match.py                           │
    │       └── Fallback: EasyOCR (vision/ocr.py) → text match         │
    │                                                                   │
    ├── WHAT ──────────────────────────────────────────────────────────┤
    │   └── YOLOv8n → class + position                                  │
    │                                                                   │
    ├── READ ──────────────────────────────────────────────────────────┤
    │   └── EasyOCR → extract text                                      │
    │                                                                   │
    └── DETAILS ───────────────────────────────────────────────────────┤
        └── Gemini Vision API (vision/gemini.py)                        │
                                                                        │
IoU Tracker (logic/tracker.py) ◄────────────────────────────────────────┘
    │
    ▼
Direction Engine (logic/direction.py)
    ├── MiDaS Depth Estimator (vision/detector.py)
    └── Horizontal position → spoken cue ("on your left — almost there")
    │
    ▼
TTS (utils/tts.py)
    ├── ElevenLabs (primary)
    └── edge-tts (fallback)
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/athulya-anil/wheredamilk.git
cd wheredamilk
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

> EasyOCR (~70 MB) and MiDaS (~400 MB) model weights download automatically on first run.

```bash
# macOS mic support (if SpeechRecognition fails)
brew install portaudio && pip install pyaudio
```

### 2. Configure API keys

Create a `.env` file in the project root:

```
# Required for DETAILS mode
GEMINI_API_KEY=your_key_here

# Optional — premium voice (falls back to edge-tts if not set)
ELEVEN_API_KEY=sk_your_key_here
ELEVEN_VOICE_ID=AeRdCCKzvd23BpJoofzx  # default: Rachel
```

Get a free Gemini key at [ai.google.dev](https://ai.google.dev/) (60 req/min free tier).

### 3. Run

```bash
python server.py
```

Open **http://localhost:8000** in your browser and click **"Start App"** to launch the vision app.

Press `q` in the OpenCV window or click **"Stop App"** to shut it down.

---

## Voice Commands

| Say | Action |
|---|---|
| `"find [item]"` | Start FIND mode |
| `"what is this"` / `"what does this say"` | Start WHAT mode |
| `"read"` / `"read this"` | Start READ mode |
| `"tell me more"` / `"tell me more about this product"` | Start DETAILS mode |
| `"stop"` / `"cancel"` | Return to idle |
| `"quit"` / `"exit"` | Close app |

---

## Project Structure

```
wheredamilk/
├── .env                  ← API keys (gitignored)
├── server.py             ← HTTP server + launcher (run this)
├── main.py               ← Vision loop (launched by server.py)
├── index.html            ← Frontend UI
│
├── vision/
│   ├── yolo.py           ← YOLOv8n detector
│   ├── ocr.py            ← EasyOCR wrapper
│   ├── detector.py       ← Detection pipeline
│   └── gemini.py         ← Google Gemini Vision API
│
├── logic/
│   ├── modes.py          ← Mode handlers (FIND, WHAT, READ, DETAILS)
│   ├── direction.py      ← Spatial direction + MiDaS depth guidance
│   ├── match.py          ← Keyword matching
│   └── tracker.py        ← IoU single-target tracker
│
└── utils/
    ├── tts.py            ← ElevenLabs + edge-tts (throttled, queue-based)
    └── speech.py         ← Continuous mic listener
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8n object detection |
| `opencv-python` | Webcam capture + drawing |
| `easyocr` | Text recognition |
| `torch` + `transformers` + `timm` | MiDaS monocular depth estimation |
| `elevenlabs` | Premium TTS (optional) |
| `edge-tts` | Fallback TTS (no API key needed) |
| `SpeechRecognition` | Voice command input |
| `python-dotenv` | `.env` key loading |

---

## Team

Built by [Athulya Anil](https://github.com/athulya-anil), [Balachandra DS](https://github.com/Baluds), and [Allen Joe Winny](https://github.com/allenjoewinny)

Created for [Hack(H)er413 2026](https://www.hackher413.com/)

🏆 **Best DEI Hack** at [Hack(H)er413 2026](https://www.hackher413.com/) — Diversity, Equity, and Inclusion Award

[Devpost Submission](https://devpost.com/software/wheredamilk)
