---
title: Mock Conference Presentation
emoji: 🎤
colorFrom: teal
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AI-Based Mock Conference Presentation System

Record a practice talk and get a score, a second-by-second timeline of pauses and
filler words, and tips for the next run.

## What it measures

| Part | How |
|---|---|
| Fluency | filler words, pauses (from word timestamps) |
| Speaking pace | words per minute while talking |
| Voice | pitch variety and volume swing (`librosa`) |
| Structure | introduction, linking words, conclusion |
| Topic focus | sentence embeddings (`sentence-transformers`) |
| Clarity and vocabulary | sentence length, repetition, word variety |
| Time management | talk length against your target |

Speech to text uses `faster-whisper`. Everything runs on the server, with no paid API.
Only English is switched on for now.

## Run it on your computer

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python web_app.py
```

The page opens at http://127.0.0.1:8000. Allow the microphone when the browser asks.
On Windows you can also double-click `start_web.bat`.

The first analysis downloads two models (speech recognition and sentence embeddings).
After that it works offline.

## Command-line version

```bash
python main.py                  # record, transcribe, score
python main.py --skip-record    # reuse the last recording
python main.py --score-only     # reuse recording and transcript
```

## Project layout

```
web_app.py          web server and API
pipeline.py         audio conversion, transcription, scoring, saved sessions
languages.py        per-language word lists (only English enabled)
speech_to_text.py   transcription settings (also a command-line script)
delivery.py         pace, fillers, pauses
audio_features.py   pitch and volume
content_analysis.py structure, focus, clarity, vocabulary, time
scoring_full.py     weighted score and tips
static/index.html   the web page
main.py             command-line version (record, transcribe, score)
speech_recorder.py  microphone recorder for the command-line version
requirements.txt    packages for running on your computer
requirements-deploy.txt  packages used by the Dockerfile
Dockerfile          used by Hugging Face Spaces
sync-to-hf.yml      optional GitHub Action: copy it to .github/workflows/
```

## Limits

- Scores come from simple rules and thresholds, not a trained grader. Treat them as
  practice feedback, not a verdict.
- Speech recognition mistakes (a misheard word) can change clarity and focus scores.
- Analysis runs on the CPU, so a few minutes of audio takes a few minutes to process.

## Privacy

On a public server each browser only sees its own sessions, and uploaded audio is
deleted after analysis. When run locally, history is shared and recordings are kept in
`output/`.
