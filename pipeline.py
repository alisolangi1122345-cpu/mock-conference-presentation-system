import json
import os
import sqlite3
import threading
import time
import traceback
import uuid
import wave
from datetime import datetime

import av
import numpy as np
from faster_whisper import WhisperModel

import content_analysis
from audio_features import audio_metrics
from content_analysis import content_metrics
from delivery import LONG_PAUSE, SHORT_PAUSE, SOFT_FILLER_GAP, delivery_metrics
from languages import clean_word, get_language
from scoring_full import build_report
from speech_to_text import BEAM_SIZE, CPU_THREADS, MODEL_SIZE

# ==========================================
# AI-Based Mock Conference Presentation System
# Pipeline used by the web app (app.py)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "output", "sessions")
DB_FILE = os.path.join(BASE_DIR, "output", "sessions.db")

PACE_WINDOW_SEC = 15
MIN_WORDS = 5

os.makedirs(DATA_DIR, exist_ok=True)


# ---------- audio conversion ----------

def convert_to_wav(src, dst):
    """Any browser/phone audio format -> 16 kHz mono 16-bit WAV (uses PyAV)."""
    try:
        container = av.open(src)
    except Exception as exc:
        raise ValueError(
            "This file could not be read as audio. Record again, or upload a "
            "WAV, MP3, M4A or WebM file."
        ) from exc
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("This file has no audio track.")

        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

        def frames_of(result):
            if result is None:
                return []
            return result if isinstance(result, list) else [result]

        with wave.open(dst, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            for frame in container.decode(stream):
                for out in frames_of(resampler.resample(frame)):
                    wf.writeframes(out.to_ndarray().tobytes())
            for out in frames_of(resampler.resample(None)):
                wf.writeframes(out.to_ndarray().tobytes())
    finally:
        container.close()


# ---------- transcription ----------

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = WhisperModel(
                MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS
            )
        return _model


def transcribe(wav_path, cfg, on_progress):
    model = get_model()
    segments, info = model.transcribe(
        wav_path,
        beam_size=BEAM_SIZE,
        language=cfg["whisper"],
        vad_filter=True,
        word_timestamps=True,
        initial_prompt=cfg["prompt"],
    )

    words, texts = [], []
    for seg in segments:
        text = seg.text.strip()
        if text:
            texts.append(text)
        for w in seg.words or []:
            word = w.word.strip()
            if word:
                words.append(
                    {
                        "word": word,
                        "start": round(float(w.start), 2),
                        "end": round(float(w.end), 2),
                        "prob": round(float(w.probability), 2),
                    }
                )
        if info.duration:
            on_progress(min(1.0, seg.end / info.duration))

    return " ".join(texts), words, info.duration


# ---------- timeline for the talk strip ----------

def mark_fillers(words, cfg):
    """True/False per word, using the same rules as delivery.py."""
    gaps = [
        max(0.0, words[i + 1]["start"] - words[i]["end"]) for i in range(len(words) - 1)
    ]
    flags = []
    for i, w in enumerate(words):
        token = clean_word(w["word"])
        if token in cfg["hard"]:
            flags.append(True)
        elif token in cfg["soft"]:
            before = gaps[i - 1] if i > 0 else 0
            after = gaps[i] if i < len(gaps) else 0
            flags.append(bool(before > SOFT_FILLER_GAP or after > SOFT_FILLER_GAP))
        else:
            flags.append(False)
    return flags


def build_timeline(words, flags, duration):
    # Speech stretches: words joined unless there is a noticeable gap
    segments = []
    cur = [words[0]["start"], words[0]["end"]]
    for prev, nxt in zip(words, words[1:]):
        if nxt["start"] - prev["end"] > SHORT_PAUSE:
            segments.append(cur)
            cur = [nxt["start"], nxt["end"]]
        else:
            cur[1] = nxt["end"]
    segments.append(cur)

    pauses = [
        [round(a["end"], 2), round(b["start"], 2)]
        for a, b in zip(words, words[1:])
        if b["start"] - a["end"] > LONG_PAUSE
    ]

    fillers = [round(w["start"], 2) for w, f in zip(words, flags) if f]

    # Words per minute in fixed windows (shows speeding up / slowing down)
    pace = []
    t = 0.0
    while t < duration:
        end = min(t + PACE_WINDOW_SEC, duration)
        count = sum(1 for w in words if t <= w["start"] < end)
        span = max(end - t, 1)
        pace.append({"t": round(t + span / 2, 1), "wpm": round(count * 60 / span)})
        t += PACE_WINDOW_SEC

    return {
        "duration": round(duration, 2),
        "segments": [[round(a, 2), round(b, 2)] for a, b in segments],
        "pauses": pauses,
        "fillers": fillers,
        "pace": pace,
    }


# ---------- sessions database ----------

def _json_default(o):
    """Safety net: turn any stray NumPy value into a plain Python one."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
               id TEXT PRIMARY KEY,
               created TEXT NOT NULL,
               language TEXT,
               overall INTEGER,
               grade TEXT,
               duration REAL,
               result TEXT NOT NULL
           )"""
    )
    return conn


def save_session(result):
    sid = uuid.uuid4().hex[:10]
    result["id"] = sid
    with _db() as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
            (
                sid,
                result["created"],
                result["language"],
                result["overall_score"],
                result["grade"],
                result["timeline"]["duration"],
                json.dumps(result, ensure_ascii=False, default=_json_default),
            ),
        )
    return sid


def list_sessions():
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, created, language, overall, grade, duration "
            "FROM sessions ORDER BY created DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(sid):
    with _db() as conn:
        row = conn.execute("SELECT result FROM sessions WHERE id = ?", (sid,)).fetchone()
    return json.loads(row["result"]) if row else None


def delete_session(sid):
    with _db() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    return cur.rowcount > 0


# ---------- full analysis ----------

def run_analysis(audio_path, language, target_minutes, progress):
    cfg = get_language(language)

    progress(2, "Preparing audio")
    wav = os.path.splitext(audio_path)[0] + "_16k.wav"
    convert_to_wav(audio_path, wav)

    progress(6, "Transcribing your speech")
    transcript, words, duration = transcribe(
        wav, cfg, lambda frac: progress(6 + frac * 64, "Transcribing your speech")
    )
    if len(words) < MIN_WORDS:
        raise ValueError(
            "Almost no speech was detected. Check the microphone and record again, "
            "closer to the mic."
        )

    progress(70, "Measuring pitch and volume")
    audio = audio_metrics(wav)

    progress(88, "Scoring")
    content_analysis.TARGET_MINUTES = target_minutes
    delivery = delivery_metrics(words, language)
    talk_seconds = words[-1]["end"] - words[0]["start"]
    content = content_metrics(transcript, talk_seconds, language)
    report = build_report(delivery, audio, content)

    flags = mark_fillers(words, cfg)
    report["language"] = cfg["name"]
    report["created"] = datetime.now().isoformat(timespec="seconds")
    report["target_minutes"] = target_minutes
    report["transcript"] = transcript
    report["words"] = [
        {"w": w["word"], "t": w["start"], "f": f} for w, f in zip(words, flags)
    ]
    report["timeline"] = build_timeline(words, flags, max(duration, words[-1]["end"]))

    progress(98, "Saving")
    return save_session(report)


# ---------- background jobs (one at a time, the CPU is the bottleneck) ----------

JOBS = {}
_run_lock = threading.Lock()


def start_job(audio_path, language, target_minutes):
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {
        "status": "queued",
        "stage": "Waiting for the previous analysis",
        "progress": 0,
        "started": time.time(),
        "session_id": None,
        "error": None,
    }
    threading.Thread(
        target=_worker,
        args=(job_id, audio_path, language, target_minutes),
        daemon=True,
    ).start()
    return job_id


def _worker(job_id, audio_path, language, target_minutes):
    job = JOBS[job_id]

    def progress(pct, stage):
        job["progress"] = round(pct, 1)
        job["stage"] = stage

    with _run_lock:
        job["status"] = "running"
        try:
            job["session_id"] = run_analysis(audio_path, language, target_minutes, progress)
            job["progress"] = 100
            job["stage"] = "Done"
            job["status"] = "done"
        except Exception as exc:
            traceback.print_exc()
            job["status"] = "error"
            job["error"] = str(exc) or exc.__class__.__name__