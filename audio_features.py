import os

import librosa
import numpy as np

# ==========================================
# AI-Based Mock Conference Presentation System
# Phase 4: Audio Features (pitch + volume variation)
# ==========================================

AUDIO_FILE = os.path.join("recordings", "presentation.wav")
TARGET_SR = 16000


def load_audio(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return y, sr


def pitch_features(y, sr):
    """Pitch (F0) statistics over voiced frames only."""
    # Speech pitch lives in ~65-500 Hz. A narrow range plus a shorter frame is
    # several times faster than librosa's defaults and avoids high octave errors.
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=65,
        fmax=500,
        sr=sr,
        frame_length=1024,
        hop_length=256,
    )
    voiced = f0[voiced_flag & ~np.isnan(f0)]

    if len(voiced) < 10:
        return {"pitch_mean_hz": 0, "pitch_std_hz": 0, "pitch_range_semitones": 0}

    mean_f0 = float(np.mean(voiced))
    # Pitch range in semitones (5th to 95th percentile) is speaker independent
    low, high = np.percentile(voiced, [5, 95])
    semitones = 12 * np.log2(high / low)

    return {
        "pitch_mean_hz": round(mean_f0, 1),
        "pitch_std_hz": round(float(np.std(voiced)), 1),
        "pitch_range_semitones": round(float(semitones), 1),
    }


def volume_features(y):
    """Loudness variation measured on frames that contain speech."""
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)

    # Keep only frames louder than the silence floor
    speech = db[db > -40]
    if len(speech) < 10:
        return {"volume_mean_db": 0, "volume_std_db": 0}

    return {
        "volume_mean_db": round(float(np.mean(speech)), 1),
        "volume_std_db": round(float(np.std(speech)), 1),
    }


def expressiveness_label(pitch_range_st):
    if pitch_range_st < 6:
        return "Monotone"
    if pitch_range_st < 10:
        return "Moderately expressive"
    return "Expressive"


def audio_metrics(path=AUDIO_FILE):
    y, sr = load_audio(path)
    result = {}
    result.update(pitch_features(y, sr))
    result.update(volume_features(y))
    result["expressiveness"] = expressiveness_label(result["pitch_range_semitones"])
    return result


def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ {AUDIO_FILE} not found. Run speech_recorder.py first.")
        return

    print("Analysing audio (this can take a few seconds)...")
    m = audio_metrics()

    print("=" * 55)
    print("   AUDIO ANALYSIS (PITCH + VOLUME)")
    print("=" * 55)
    print(f"Average pitch        : {m['pitch_mean_hz']} Hz")
    print(f"Pitch variation (std): {m['pitch_std_hz']} Hz")
    print(f"Pitch range          : {m['pitch_range_semitones']} semitones")
    print(f"Voice style          : {m['expressiveness']}")
    print(f"Average volume       : {m['volume_mean_db']} dB (relative to peak)")
    print(f"Volume variation     : {m['volume_std_db']} dB")


if __name__ == "__main__":
    main()
