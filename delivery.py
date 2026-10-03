import json
import os

from languages import clean_word, current_language_code, get_language

# ==========================================
# AI-Based Mock Conference Presentation System
# Phase 3: Delivery Metrics (WPM, fillers, pauses) - multi-language
# ==========================================

WORDS_FILE = os.path.join("output", "presentation_words.json")

SHORT_PAUSE = 0.3   # gaps above this count toward pause time
LONG_PAUSE = 1.0    # gaps above this count as a "long pause"
SOFT_FILLER_GAP = 0.4


def count_fillers(words, gaps, cfg):
    """Hard fillers always count; soft fillers (like 'so', 'matlab') only near a pause."""
    hard, soft = 0, 0
    found = []

    for i, w in enumerate(words):
        token = clean_word(w["word"])

        if token in cfg["hard"]:
            hard += 1
            found.append(token)
        elif token in cfg["soft"]:
            before = gaps[i - 1] if i > 0 else 0
            after = gaps[i] if i < len(gaps) else 0
            if before > SOFT_FILLER_GAP or after > SOFT_FILLER_GAP:
                soft += 1
                found.append(token)

    return hard + soft, found


def pace_label(wpm):
    if wpm < 100:
        return "Too slow"
    if wpm <= 160:
        return "Good pace"
    if wpm <= 180:
        return "A bit fast"
    return "Too fast"


def delivery_metrics(words, lang=None):
    cfg = get_language(lang)

    if len(words) < 2:
        raise ValueError("Not enough words to analyse.")

    start = words[0]["start"]
    end = words[-1]["end"]
    duration = end - start

    gaps = [
        max(0.0, words[i + 1]["start"] - words[i]["end"])
        for i in range(len(words) - 1)
    ]

    pause_time = sum(g for g in gaps if g > SHORT_PAUSE)
    long_pauses = sum(1 for g in gaps if g > LONG_PAUSE)
    speaking_time = max(duration - pause_time, 1e-6)

    wpm = len(words) / (duration / 60)
    speaking_wpm = len(words) / (speaking_time / 60)

    filler_count, filler_list = count_fillers(words, gaps, cfg)

    return {
        "language": cfg["code"],
        "total_words": len(words),
        "duration_sec": round(duration, 1),
        "wpm": round(wpm),
        "speaking_wpm": round(speaking_wpm),
        "pace": pace_label(wpm),
        "filler_count": filler_count,
        "fillers_per_100_words": round(filler_count / len(words) * 100, 1),
        "fillers_found": filler_list,
        "long_pauses": long_pauses,
        "pause_ratio": round(pause_time / duration, 2),
    }


def main():
    if not os.path.exists(WORDS_FILE):
        print(f"❌ {WORDS_FILE} not found. Run speech_to_text.py first.")
        return

    with open(WORDS_FILE, encoding="utf-8") as f:
        words = json.load(f)

    code = current_language_code()
    m = delivery_metrics(words, code)

    print("=" * 55)
    print("   DELIVERY ANALYSIS")
    print("=" * 55)
    print(f"Language           : {get_language(code)['name']}")
    print(f"Total words        : {m['total_words']}")
    print(f"Duration           : {m['duration_sec']} sec")
    print(f"Speed              : {m['wpm']} WPM  ({m['pace']})")
    print(f"Speed (no pauses)  : {m['speaking_wpm']} WPM")
    print(f"Filler words       : {m['filler_count']} "
          f"({m['fillers_per_100_words']} per 100 words)")
    print(f"Fillers found      : {', '.join(m['fillers_found']) or 'none'}")
    print(f"Long pauses (>1s)  : {m['long_pauses']}")
    print(f"Pause ratio        : {m['pause_ratio'] * 100:.0f}% of the time")


if __name__ == "__main__":
    main()
