import os
import wave

import numpy as np
import sounddevice as sd

# ==========================================
# AI-Based Mock Conference Presentation System
# Phase 1: Speech Recorder (improved)
# ==========================================

PREFERRED_RATE = 16000  # Whisper uses 16 kHz internally
CHANNELS = 1

OUTPUT_DIR = "recordings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "presentation.wav")

os.makedirs(OUTPUT_DIR, exist_ok=True)

recorded_chunks = []


def audio_callback(indata, frames, time, status):
    """Collect audio data while recording."""
    if status:
        print("Audio status:", status)
    recorded_chunks.append(indata.copy())


def choose_input_device():
    """Show input devices and let the user pick one (ENTER = system default)."""
    devices = sd.query_devices()
    inputs = [
        (i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0
    ]

    if not inputs:
        raise RuntimeError("No input (microphone) device found.")

    default_in = sd.default.device[0]

    print("\nAvailable microphones:")
    for i, d in inputs:
        mark = "  <-- default" if i == default_in else ""
        print(f"  [{i}] {d['name']}{mark}")

    valid_ids = {i for i, _ in inputs}

    while True:
        choice = input("\nEnter device number (or press ENTER for default): ").strip()
        if choice == "":
            return None  # let sounddevice use the system default
        if choice.isdigit() and int(choice) in valid_ids:
            return int(choice)
        print("Invalid choice, try again.")


def pick_sample_rate(device):
    """Use 16 kHz if the mic supports it, otherwise the device's own rate."""
    try:
        sd.check_input_settings(
            device=device, channels=CHANNELS, samplerate=PREFERRED_RATE
        )
        return PREFERRED_RATE
    except Exception:
        info = sd.query_devices(device, "input")
        rate = int(info["default_samplerate"])
        print(f"16 kHz not supported, using {rate} Hz instead.")
        return rate


def main():
    print("=" * 55)
    print("   AI-BASED MOCK CONFERENCE PRESENTATION SYSTEM")
    print("=" * 55)

    device = choose_input_device()
    sample_rate = pick_sample_rate(device)

    mic_info = sd.query_devices(device, "input")
    print(f"\nUsing microphone: {mic_info['name']} @ {sample_rate} Hz")

    input("\nPress ENTER to START recording...")

    print("\n🎤 Recording started!")
    print("Give your presentation.")
    print("Press ENTER when you want to STOP.\n")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="int16",
        device=device,
        callback=audio_callback,
    ):
        input()

    print("\n✅ Recording stopped!")

    if not recorded_chunks:
        print("❌ No audio was recorded.")
        return

    audio = np.concatenate(recorded_chunks, axis=0)

    peak = int(np.max(np.abs(audio)))
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))

    print(f"Peak level : {peak}")
    print(f"RMS level  : {rms:.2f}")

    with wave.open(OUTPUT_FILE, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    duration = len(audio) / sample_rate

    print(f"\n💾 Recording saved: {OUTPUT_FILE}")
    print(f"⏱️ Recording duration: {duration:.2f} seconds")

    if peak < 100:
        print("⚠️ Warning: Audio seems very quiet. Check your microphone.")
    elif peak > 32000:
        print("⚠️ Warning: Audio may be clipping (too loud). Move back a bit.")
    else:
        print("✅ Audio captured successfully!")


if __name__ == "__main__":
    main()