import json
import os
import sys
import time

from faster_whisper import WhisperModel

from languages import (
    available_languages,
    current_language_code,
    get_language,
    save_language,
)

# ==========================================
# AI-Based Mock Conference Presentation System
# Phase 2: Speech-to-Text (multi-language)
#
# Usage:
#   python speech_to_text.py          -> English (only language enabled for now)
# More languages: add them to ENABLED_LANGUAGES in languages.py, then
#   python speech_to_text.py          -> asks which language
#   python speech_to_text.py ur       -> Urdu (en, ur, hi, ar)
# ==========================================

AUDIO_FILE = os.path.join("recordings", "presentation.wav")

OUTPUT_DIR = "output"
TRANSCRIPT_FILE = os.path.join(OUTPUT_DIR, "presentation_transcript.txt")
WORDS_FILE = os.path.join(OUTPUT_DIR, "presentation_words.json")

# "small" is fast on CPU. For Urdu / Hindi / Arabic, "medium" is noticeably
# more accurate (but slower and ~1.5 GB download). Try it if results are poor.
MODEL_SIZE = "small"

# beam_size 5 is more accurate; 1 (greedy) is roughly 2x faster.
BEAM_SIZE = 5

# Use all CPU cores for transcription
CPU_THREADS = os.cpu_count() or 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


def choose_language():
    """Language from command line, otherwise a menu (ENTER keeps the last one)."""
    codes = [c for c, _ in available_languages()]

    if len(codes) == 1:  # only one language enabled: no need to ask
        return codes[0]

    if len(sys.argv) > 1:
        code = sys.argv[1].strip().lower()
        if code not in codes:
            print(f"Unknown language '{code}'. Choose from: {', '.join(codes)}")
            sys.exit(1)
        return code

    last = current_language_code()
    print("\nSelect the language of the presentation:")
    for i, (code, name) in enumerate(available_languages(), 1):
        mark = "  <-- last used" if code == last else ""
        print(f"  [{i}] {name} ({code}){mark}")

    while True:
        choice = input(f"\nNumber (ENTER = {last}): ").strip()
        if choice == "":
            return last
        if choice.isdigit() and 1 <= int(choice) <= len(codes):
            return codes[int(choice) - 1]
        print("Invalid choice, try again.")


def main():
    print("=" * 55)
    print("   AI-BASED MOCK CONFERENCE PRESENTATION SYSTEM")
    print("=" * 55)

    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Audio file not found: {AUDIO_FILE}")
        print("Run speech_recorder.py first.")
        return

    code = choose_language()
    save_language(code)  # other scripts read this file
    cfg = get_language(code)
    print(f"\nLanguage: {cfg['name']}")

    print("\nLoading speech recognition model...")
    print("Please wait...")

    t_load = time.time()
    model = WhisperModel(
        MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS
    )

    print(f"✅ Model loaded successfully! ({time.time() - t_load:.1f}s)")
    print("\nTranscribing presentation...")

    t_trans = time.time()
    segments, info = model.transcribe(
        AUDIO_FILE,
        beam_size=BEAM_SIZE,
        language=cfg["whisper"],
        vad_filter=True,
        word_timestamps=True,
        initial_prompt=cfg["prompt"],
    )

    print(f"Audio duration: {info.duration:.2f} seconds\n")

    words = []
    full_text = []

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}")
        full_text.append(text)

        for w in segment.words or []:
            word = w.word.strip()
            if word:
                words.append(
                    {
                        "word": word,
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "prob": round(w.probability, 2),
                    }
                )

    transcript = " ".join(full_text)
    trans_time = time.time() - t_trans

    print("\n" + "=" * 55)
    print("FINAL TRANSCRIPT")
    print("=" * 55)
    print("\n" + transcript)

    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write(transcript)

    with open(WORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Transcript saved to: {TRANSCRIPT_FILE}")
    print(f"✅ {len(words)} words with timestamps saved to: {WORDS_FILE}")
    print(f"⏱️ Transcription took {trans_time:.0f}s for {info.duration:.0f}s of audio")


if __name__ == "__main__":
    main()
