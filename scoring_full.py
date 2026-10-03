import json
import os

from audio_features import AUDIO_FILE, audio_metrics
from content_analysis import TRANSCRIPT_FILE, WORDS_FILE, content_metrics
from delivery import delivery_metrics
from languages import current_language_code, get_language

# ==========================================
# AI-Based Mock Conference Presentation System
# Phase 6: Scoring Engine (weighted rubric + tips)
# ==========================================

OUTPUT_DIR = "output"
REPORT_FILE = os.path.join(OUTPUT_DIR, "final_report.json")

# Weights add up to 100. Change them to suit your rubric.
WEIGHTS = {
    "fluency": 20,
    "pace": 15,
    "voice": 10,
    "structure": 15,
    "topic_focus": 15,
    "clarity": 10,
    "vocabulary": 5,
    "time_management": 10,
}

LABELS = {
    "fluency": "Fluency (fillers, pauses)",
    "pace": "Speaking pace",
    "voice": "Voice (pitch, volume)",
    "structure": "Structure",
    "topic_focus": "Topic focus",
    "clarity": "Clarity",
    "vocabulary": "Vocabulary",
    "time_management": "Time management",
}


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def linear(value, bad, good):
    """Map value to 0-100: 'bad' -> 0, 'good' -> 100 (works in either direction)."""
    return clamp((value - bad) / (good - bad) * 100)


# ---------- category scorers (return score, tips) ----------

def score_fluency(d):
    filler_score = linear(d["fillers_per_100_words"], 8, 1)
    pause_score = linear(d["pause_ratio"], 0.5, 0.15)
    long_per_min = d["long_pauses"] / max(d["duration_sec"] / 60, 0.5)
    long_pause_score = linear(long_per_min, 6, 1)
    score = 0.5 * filler_score + 0.3 * pause_score + 0.2 * long_pause_score

    tips = []
    if filler_score < 70:
        found = ", ".join(sorted(set(d["fillers_found"]))) or "fillers"
        tips.append(f"Reduce filler words ({found}). Pause silently instead.")
    if pause_score < 60 or long_pause_score < 60:
        tips.append("Too many long pauses. Practise your script until it flows.")
    return score, tips


def score_pace(d):
    wpm = d["speaking_wpm"]  # pauses excluded; pauses are already scored in fluency
    if 110 <= wpm <= 160:
        score = 100
    elif wpm < 110:
        score = linear(wpm, 50, 110)
    else:
        score = linear(wpm, 220, 160)

    tips = []
    if wpm < 110:
        tips.append(f"You speak slowly ({wpm} WPM). Aim for 110-160 WPM.")
    elif wpm > 160:
        tips.append(f"You speak fast ({wpm} WPM). Slow down so the audience can follow.")
    return score, tips


def score_voice(a):
    pitch_score = linear(a["pitch_range_semitones"], 3, 10)
    if 3 <= a["volume_std_db"] <= 9:
        volume_score = 100
    else:
        volume_score = linear(abs(a["volume_std_db"] - 6), 12, 3)
    score = 0.7 * pitch_score + 0.3 * volume_score

    tips = []
    if pitch_score < 60:
        tips.append("Your voice sounds flat. Vary your pitch and stress key words.")
    if a["volume_std_db"] > 9:
        tips.append("Your volume changes a lot. Keep your loudness more even.")
    return score, tips


# ---------- main scoring ----------

def grade(score):
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Fair"
    return "Needs improvement"


def build_report(delivery, audio, content):
    scores, tips = {}, {}

    scores["fluency"], tips["fluency"] = score_fluency(delivery)
    scores["pace"], tips["pace"] = score_pace(delivery)
    scores["voice"], tips["voice"] = score_voice(audio)

    for key in ("structure", "topic_focus", "clarity", "vocabulary", "time_management"):
        r = content[key]
        scores[key] = r.get("score")
        tips[key] = list(r.get("tips", []))
        if key == "time_management" and r.get("tip") and (r.get("score") or 0) < 100:
            tips[key].append(r["tip"])

    # Skip categories that could not be judged and re-balance the weights
    available = {k: v for k, v in scores.items() if v is not None}
    total_weight = sum(WEIGHTS[k] for k in available)
    overall = sum(available[k] * WEIGHTS[k] for k in available) / total_weight

    not_judged = [LABELS[k] for k, v in scores.items() if v is None]

    breakdown = {
        k: {
            "label": LABELS[k],
            "weight": WEIGHTS[k],
            "score": None if scores[k] is None else round(scores[k]),
        }
        for k in WEIGHTS
    }

    # Tips from the weakest categories first
    top_tips = []
    for k in sorted(available, key=lambda k: available[k]):
        for t in tips[k]:
            if t not in top_tips:
                top_tips.append(t)

    return {
        "overall_score": round(overall),
        "grade": grade(overall),
        "breakdown": breakdown,
        "not_judged": not_judged,
        "tips": top_tips[:5],
        "raw": {"delivery": delivery, "audio": audio, "content": content},
    }


def run_pipeline():
    for path in (WORDS_FILE, TRANSCRIPT_FILE, AUDIO_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run speech_recorder.py and speech_to_text.py first."
            )

    with open(WORDS_FILE, encoding="utf-8") as f:
        words = json.load(f)
    with open(TRANSCRIPT_FILE, encoding="utf-8") as f:
        transcript = f.read()

    duration = words[-1]["end"] - words[0]["start"]

    lang = current_language_code()
    delivery = delivery_metrics(words, lang)
    print("Analysing audio (pitch can take a few seconds)...")
    audio = audio_metrics(AUDIO_FILE)
    content = content_metrics(transcript, duration, lang)

    report = build_report(delivery, audio, content)
    report["language"] = get_language(lang)["name"]
    return report


def print_report(r):
    print("\n" + "=" * 55)
    print("   FINAL PRESENTATION REPORT")
    print("=" * 55)
    print(f"Language: {r.get('language', 'English')}")
    print(f"\nOVERALL SCORE: {r['overall_score']}/100  ({r['grade']})\n")

    print(f"{'Category':<28}{'Weight':>7}{'Score':>8}")
    print("-" * 43)
    for b in r["breakdown"].values():
        shown = "n/a" if b["score"] is None else f"{b['score']}"
        print(f"{b['label']:<28}{str(b['weight']) + '%':>7}{shown:>8}")

    if r["not_judged"]:
        print(f"\nNot judged (recording too short): {', '.join(r['not_judged'])}")

    if r["tips"]:
        print("\nTOP IMPROVEMENT TIPS")
        for i, t in enumerate(r["tips"], 1):
            print(f"  {i}. {t}")


def main():
    report = run_pipeline()
    print_report(report)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Full report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
