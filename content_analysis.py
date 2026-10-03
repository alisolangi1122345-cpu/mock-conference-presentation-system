import json
import math
import os
from collections import Counter

import numpy as np

from languages import (
    current_language_code,
    get_language,
    has_cue,
    split_sentences,
    tokenize,
)

# ==========================================
# AI-Based Mock Conference Presentation System
# Phase 5: Content Analysis (offline, transcript only) - multi-language
# ==========================================

OUTPUT_DIR = "output"
TRANSCRIPT_FILE = os.path.join(OUTPUT_DIR, "presentation_transcript.txt")
WORDS_FILE = os.path.join(OUTPUT_DIR, "presentation_words.json")
RESULT_FILE = os.path.join(OUTPUT_DIR, "content_analysis.json")

# Time management settings (change to your conference slot)
TARGET_MINUTES = 3.0
TOLERANCE = 0.25  # +/-25% of target is still full marks

# Neural model for topic focus (falls back to word-overlap if unavailable)
try:
    from sentence_transformers import SentenceTransformer
    HAVE_EMBEDDINGS = True
except ImportError:
    HAVE_EMBEDDINGS = False

_MODELS = {}


# ---------- helpers ----------

def content_words(text, cfg):
    return [
        w for w in tokenize(text)
        if w not in cfg["stopwords"] and len(w) >= cfg["min_len"]
    ]


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


# ---------- 1. structure ----------

def analyse_structure(sentences, cfg):
    n = len(sentences)
    if n < 3:
        return {"score": None, "note": "Too short to judge structure (need 3+ sentences)."}

    head = " ".join(sentences[: max(1, math.ceil(n * 0.25))])
    tail = " ".join(sentences[-max(1, math.ceil(n * 0.25)):])
    middle = " ".join(sentences[1:-1])

    intro = has_cue(head, cfg["intro"])
    conclusion = has_cue(tail, cfg["conclusion"])
    transitions = sum(1 for c in cfg["transitions"] if has_cue(middle, [c]))

    score = 0
    tips = []
    if intro:
        score += 35
    else:
        tips.append("Start with a clear introduction: greet, say your name and topic.")
    if conclusion:
        score += 35
    else:
        tips.append("End with a conclusion or summary before saying thank you.")
    if n >= 6:
        score += min(30, transitions * 10)
        if transitions < 2:
            tips.append("Use linking words (first, next, because, for example) to connect points.")
    else:
        score += 15
        tips.append("Add more body content: explain 2-3 main points with examples.")

    return {
        "score": round(clamp(score)),
        "has_intro": intro,
        "has_conclusion": conclusion,
        "transition_words": transitions,
        "tips": tips,
    }


# ---------- 2. topic focus ----------

def _bow_vectors(sentences, cfg):
    docs = [content_words(s, cfg) for s in sentences]
    vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
    mat = np.zeros((len(sentences), max(1, len(vocab))))
    for r, d in enumerate(docs):
        for w in d:
            mat[r, vocab[w]] += 1
    return mat


def _embed_vectors(sentences, model_name):
    if model_name not in _MODELS:
        print(f"Loading sentence model '{model_name}' (first time needs internet)...")
        _MODELS[model_name] = SentenceTransformer(model_name)
    return np.asarray(_MODELS[model_name].encode(sentences))


def analyse_focus(sentences, cfg):
    if len(sentences) < 5:
        return {"score": None, "note": "Too short to judge topic focus (need 5+ sentences)."}

    method = "word-overlap"
    vecs = None
    if HAVE_EMBEDDINGS and cfg["embed_model"]:
        try:
            vecs = _embed_vectors(sentences, cfg["embed_model"])
            method = "sentence-embeddings"
        except Exception as exc:  # no internet on first run, etc.
            print(f"(Embedding model unavailable: {exc}) Using word-overlap instead.")
    if vecs is None:
        vecs = _bow_vectors(sentences, cfg)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit = vecs / norms

    centroid = unit.mean(axis=0)
    c_norm = np.linalg.norm(centroid)
    if c_norm == 0:
        return {"score": 0, "method": method, "off_topic_sentences": []}
    centroid = centroid / c_norm

    sims = unit @ centroid
    mean_sim = float(np.mean(sims))

    # Different methods give different similarity scales
    lo, hi = (0.25, 0.65) if method == "sentence-embeddings" else (0.05, 0.45)
    score = clamp((mean_sim - lo) / (hi - lo) * 100)

    # Ignore first and last sentence (greeting / thank-you are naturally generic)
    off = [
        sentences[i]
        for i, s in enumerate(sims)
        if 0 < i < len(sims) - 1 and s < lo
    ]

    return {
        "score": round(score),
        "method": method,
        "mean_similarity": round(mean_sim, 2),
        "off_topic_sentences": off[:3],
    }


# ---------- 3. clarity ----------

def analyse_clarity(sentences, words, cfg):
    if not sentences:
        return {"score": None, "note": "No sentences found."}

    lengths = [len(s.split()) for s in sentences]
    avg_len = float(np.mean(lengths))

    stop = cfg["stopwords"]
    repeats = sum(1 for a, b in zip(words, words[1:]) if a == b and a not in stop)
    trigrams = Counter(zip(words, words[1:], words[2:]))
    repeated_phrases = sum(c - 1 for c in trigrams.values() if c > 1)

    score = 100
    tips = []
    if avg_len < 6:
        score -= 20
        tips.append("Sentences are very short or broken. Try to complete your thoughts.")
    elif avg_len > 28:
        score -= 25
        tips.append("Sentences are very long. Break them into shorter ones.")

    score -= min(25, repeats * 8)
    score -= min(25, repeated_phrases * 8)
    if repeats or repeated_phrases:
        tips.append("You repeated some words or phrases. Plan your sentences to avoid this.")

    return {
        "score": round(clamp(score)),
        "avg_sentence_words": round(avg_len, 1),
        "repeated_words": repeats,
        "repeated_phrases": repeated_phrases,
        "tips": tips,
    }


# ---------- 4. vocabulary ----------

def analyse_vocabulary(words, cfg):
    cw = [w for w in words if w not in cfg["stopwords"] and len(w) >= cfg["min_len"]]
    if len(cw) < 30:
        return {"score": None, "note": "Too short to judge vocabulary (need ~30+ content words)."}

    unique = len(set(cw))
    # Guiraud's root TTR is less length-dependent than plain unique/total
    guiraud = unique / math.sqrt(len(cw))
    score = clamp((guiraud - 4.0) / (9.0 - 4.0) * 100)

    return {
        "score": round(score),
        "unique_content_words": unique,
        "total_content_words": len(cw),
        "diversity_index": round(guiraud, 2),
    }


# ---------- 5. time management ----------

def analyse_time(duration_sec):
    if not duration_sec:
        return {"score": None, "note": "Duration unknown."}

    target = TARGET_MINUTES * 60
    unit = "minute" if TARGET_MINUTES == 1 else "minutes"
    ratio = duration_sec / target
    low, high = 1 - TOLERANCE, 1 + TOLERANCE

    if low <= ratio <= high:
        score, tip = 100, "Great timing."
    elif ratio < low:
        score = clamp(100 - (low - ratio) * 150)
        tip = f"Too short. Aim for about {TARGET_MINUTES:g} {unit}."
    else:
        score = clamp(100 - (ratio - high) * 150)
        tip = f"Too long. Aim for about {TARGET_MINUTES:g} {unit}."

    return {
        "score": round(score),
        "duration_sec": round(duration_sec, 1),
        "target_sec": round(target),
        "tip": tip,
    }


# ---------- main entry ----------

def content_metrics(transcript, duration_sec=None, lang=None):
    cfg = get_language(lang)
    sentences = split_sentences(transcript)
    words = tokenize(transcript)

    return {
        "language": cfg["code"],
        "structure": analyse_structure(sentences, cfg),
        "topic_focus": analyse_focus(sentences, cfg),
        "clarity": analyse_clarity(sentences, words, cfg),
        "vocabulary": analyse_vocabulary(words, cfg),
        "time_management": analyse_time(duration_sec),
    }


def main():
    if not os.path.exists(TRANSCRIPT_FILE):
        print(f"❌ {TRANSCRIPT_FILE} not found. Run speech_to_text.py first.")
        return

    with open(TRANSCRIPT_FILE, encoding="utf-8") as f:
        transcript = f.read()

    duration = None
    if os.path.exists(WORDS_FILE):
        with open(WORDS_FILE, encoding="utf-8") as f:
            words = json.load(f)
        if words:
            duration = words[-1]["end"] - words[0]["start"]

    code = current_language_code()
    m = content_metrics(transcript, duration, code)

    print("=" * 55)
    print("   CONTENT ANALYSIS (OFFLINE)")
    print("=" * 55)
    print(f"Language: {get_language(code)['name']}")

    labels = {
        "structure": "Structure",
        "topic_focus": "Topic focus",
        "clarity": "Clarity",
        "vocabulary": "Vocabulary",
        "time_management": "Time management",
    }

    for key, label in labels.items():
        r = m[key]
        score = r.get("score")
        shown = "n/a" if score is None else f"{score}/100"
        print(f"\n{label}: {shown}")
        if r.get("note"):
            print(f"  - {r['note']}")
        for tip in r.get("tips", []):
            print(f"  - {tip}")
        if key == "time_management" and r.get("tip"):
            print(f"  - {r['tip']}")
        if key == "topic_focus" and r.get("off_topic_sentences"):
            print("  - Possibly off-topic:")
            for s in r["off_topic_sentences"]:
                print(f"      \"{s}\"")

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
