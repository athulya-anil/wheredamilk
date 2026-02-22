"""
utils/tts.py — Queued TTS using ElevenLabs with pyttsx3 fallback.

Set your API key via environment variable:
    export ELEVEN_API_KEY="sk-..."

Features:
  1. Queue-based TTS: all requests are processed sequentially
  2. No messages missed due to concurrent requests
  3. Throttled speak() for continuous updates (skips if text hasn't changed)
  4. speak_once() for important messages (always queued)
  5. Background worker thread handles all speech delivery
  6. Falls back to pyttsx3 on ElevenLabs errors
"""

import os
import io
import time
import threading
import queue as queue_module

THROTTLE_SECS = 1.0

# ── ElevenLabs setup ──────────────────────────────────────────────────────────
_ELEVEN_KEY = os.environ.get("ELEVEN_API_KEY", "")

try:
    from elevenlabs.client import ElevenLabs
    _ELEVEN_AVAILABLE = bool(_ELEVEN_KEY)
except ImportError:
    _ELEVEN_AVAILABLE = False

# Try to import audio playback (fallback to write-to-file if unavailable)
try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    _PYDUB_AVAILABLE = True
except ImportError:
    _PYDUB_AVAILABLE = False
    try:
        import soundfile as sf
        import sounddevice as sd
        _SOUNDDEVICE_AVAILABLE = True
    except ImportError:
        _SOUNDDEVICE_AVAILABLE = False

# Try to import edge-tts as fallback TTS engine (thread-safe, no API key needed)
try:
    import edge_tts
    _EDGE_TTS_AVAILABLE = True
except ImportError:
    _EDGE_TTS_AVAILABLE = False

# Async helper for edge-tts (runs in thread pool to avoid blocking)
import asyncio
try:
    from concurrent.futures import ThreadPoolExecutor
    _EXECUTOR = ThreadPoolExecutor(max_workers=1)
except ImportError:
    _EXECUTOR = None

# ElevenLabs settings — tweak to taste
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "AeRdCCKzvd23BpJoofzx")
ELEVEN_MODEL    = "eleven_turbo_v2"   # lowest-latency model (~300 ms)

class TTSEngine:
    def __init__(self):
        self._queue = queue_module.Queue()
        self._last_text: str  = ""
        self._last_time: float = 0.0
        self._stop_event = threading.Event()
        self._client = None
                
        if _ELEVEN_AVAILABLE:
            try:
                self._client = ElevenLabs(api_key=_ELEVEN_KEY)
                print(f"[tts] ElevenLabs ready  (voice={ELEVEN_VOICE_ID}, model={ELEVEN_MODEL})")
            except Exception as e:
                print(f"[tts] Failed to initialize ElevenLabs: {e}")
                self._client = None
        else:
            print("[tts] ElevenLabs not available (key missing or package not installed).")
        
        # Start background worker thread
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()

    def reset_throttle(self) -> None:
        """Reset throttle state. Call when mode changes to avoid blocking repeated announcements."""
        self._last_text = ""
        self._last_time = 0.0

    def speak(self, text: str) -> None:
        """Throttled speak — skips if text unchanged within THROTTLE_SECS.
        Non-blocking: queues request and returns immediately.
        
        Use for continuous updates (e.g., directions "left", "left", "left")
        Use speak_once() for important one-time announcements instead.
        """
        text = text.strip()
        if not text:
            return
        
        now = time.time()
        if text == self._last_text and (now - self._last_time) < THROTTLE_SECS:
            return  # Skip throttled message
        
        self._last_text = text
        self._last_time = now
        queue_size = self._queue.qsize()
        self._queue.put(("speak", text))


    def speak_once(self, text: str) -> None:
        """Important message — always queued, not throttled.
        Non-blocking: queues request and returns immediately.
        """
        text = text.strip()
        if text:
            queue_size = self._queue.qsize()
            self._queue.put(("speak_once", text))

    def stop(self) -> None:
        """Signal worker to stop processing."""
        self._stop_event.set()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_queue(self) -> None:
        """Background worker: process TTS requests sequentially."""
        iterations = 0
        while not self._stop_event.is_set():
            iterations += 1
            try:
                msg_type, text = self._queue.get(timeout=2.0)
                self._say(text)
                time.sleep(0.3)
            except queue_module.Empty:
                if iterations % 20 == 0:
                    print(f"[tts] ⏳ Worker idle...")
                continue
            except Exception as e:
                print(f"[tts] Queue error: {e}")

    def _say(self, text: str) -> None:
        """Execute TTS via ElevenLabs or edge-tts fallback."""
        print(f"[tts] _say() called with: '{text}'")
        
        if self._client and _ELEVEN_AVAILABLE:
            try:
                audio_bytes = self._client.text_to_speech.convert(
                    voice_id=ELEVEN_VOICE_ID,
                    text=text,
                    model_id=ELEVEN_MODEL,
                )
                self._play_audio(audio_bytes)
                return
            except Exception as exc:
                print(f"[tts] ElevenLabs error ({type(exc).__name__}): {exc}")
        
        self._say_edge_tts(text)

    def _say_edge_tts(self, text: str) -> None:
        """Fallback TTS using edge-tts (cloud-based, thread-safe).
        
        edge-tts is Microsoft's TTS engine—high quality, no API key needed.
        It's async, so we run it via asyncio event loop.
        """
        if not _EDGE_TTS_AVAILABLE:
            print(f"[tts] edge-tts not available. Install with: pip install edge-tts")
            return
        
        try:            
            # Run async function in a new event loop (safe for worker thread)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                audio_bytes = loop.run_until_complete(self._edge_tts_get_audio(text))
                if audio_bytes:
                    self._play_audio(audio_bytes)
                else:
                    print(f"[tts] edge-tts generated no audio")
            finally:
                loop.close()
            
        except Exception as e:
            print(f"[tts] edge-tts error ({type(e).__name__}): {e}")
    
    async def _edge_tts_get_audio(self, text: str) -> bytes:
        """Async helper to get audio from edge-tts."""
        try:
            communicate = edge_tts.Communicate(text=text, voice="en-US-AriaNeural")
            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
            return b"".join(audio_chunks)
        except Exception as e:
            print(f"[tts] edge-tts async error: {e}")
            return None

    def _play_audio(self, audio_data) -> None:
        """Play audio bytes using available backend. Handles both bytes and generators."""
        # Convert generator to bytes if needed
        if hasattr(audio_data, '__iter__') and not isinstance(audio_data, (bytes, bytearray)):
            try:
                audio_bytes = b"".join(audio_data)
            except Exception as e:
                print(f"[tts] Failed to convert audio stream: {e}")
                return
        else:
            audio_bytes = audio_data
        
        if _PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
                pydub_play(audio)
                return
            except Exception as e:
                print(f"[tts] pydub playback failed: {e}")
        
        if _SOUNDDEVICE_AVAILABLE:
            try:
                import soundfile as sf
                import sounddevice as sd
                with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                    sd.play(f.read(), f.samplerate)
                    sd.wait()
                return
            except Exception as e:
                print(f"[tts] sounddevice playback failed: {e}")
        
        # Fallback: save to temp file
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            os.system(f"afplay '{temp_path}'")  # macOS
            os.remove(temp_path)
        except Exception as e:
            print(f"[tts] No audio playback available: {e}")
