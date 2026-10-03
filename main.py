import argparse
import os
import subprocess
import sys
import time

# ==========================================
# AI-Based Mock Conference Presentation System
# main.py: runs the whole pipeline with one command
#
#   python main.py                 record -> transcribe -> score
#   python main.py --skip-record   reuse the existing recording
#   python main.py --score-only    reuse recording AND transcript
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_FILE = os.path.join("recordings", "presentation.wav")
WORDS_FILE = os.path.join("output", "presentation_words.json")
TRANSCRIPT_FILE = os.path.join("output", "presentation_transcript.txt")

STEPS = [
    ("record", "Step 1/3: Recording", "speech_recorder.py"),
    ("transcribe", "Step 2/3: Speech to text", "speech_to_text.py"),
    ("score", "Step 3/3: Analysis and scoring", "scoring_full.py"),
]


def banner(text):
    print("\n" + "=" * 55)
    print(f"  {text}")
    print("=" * 55 + "\n")


def run_script(script):
    """Run a script with the same Python (so the venv is used) from the project folder."""
    result = subprocess.run([sys.executable, script], cwd=BASE_DIR)
    return result.returncode == 0


def need(path, hint):
    full = os.path.join(BASE_DIR, path)
    if not os.path.exists(full):
        print(f"❌ {path} not found. {hint}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="AI-Based Mock Conference Presentation System")
    parser.add_argument("--skip-record", action="store_true",
                        help="use the existing recordings/presentation.wav")
    parser.add_argument("--score-only", action="store_true",
                        help="use the existing recording and transcript, only re-run scoring")
    args = parser.parse_args()

    skip = set()
    if args.skip_record or args.score_only:
        skip.add("record")
    if args.score_only:
        skip.add("transcribe")

    # Make sure the files a skipped step would have made are there
    if "record" in skip and not need(AUDIO_FILE, "Run without --skip-record first."):
        return 1
    if "transcribe" in skip and not (
        need(WORDS_FILE, "Run without --score-only first.")
        and need(TRANSCRIPT_FILE, "Run without --score-only first.")
    ):
        return 1

    banner("AI-BASED MOCK CONFERENCE PRESENTATION SYSTEM")
    started = time.time()
    timings = []

    for key, title, script in STEPS:
        if key in skip:
            print(f"(skipping {script})")
            continue

        banner(title)
        step_start = time.time()
        ok = run_script(script)
        timings.append((script, time.time() - step_start))
        if not ok:
            print(f"\n❌ {script} failed. Fix the error above, then run main.py again.")
            if key != "record":
                print("Tip: use --skip-record to keep your existing recording.")
            return 1

    print("\nTime per step:")
    for script, secs in timings:
        print(f"  {script:<20}{secs:6.0f} s")

    minutes = (time.time() - started) / 60
    print(f"\n✅ Done in {minutes:.1f} min. Full report: output\\final_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
