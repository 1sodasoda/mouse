"""Guided mouse-data collector.

Prompts you to perform a gesture, records while you do it, and saves the sample
into data/raw/<gesture>/. Then it prompts the next one and repeats.

Run from the src/ directory:  python data_collector.py

Flow per sample:
  1. It names the gesture + instruction.
  2. Press Enter to START recording.
  3. Perform the movement.
  4. Press Enter to STOP — the sample is saved, then it moves to the next.

Ctrl+C to quit at any time.
"""
import os
import datetime
import itertools

from logger.mouse_logger import Logger
from etc import tui

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(HERE, "..", "data", "raw")

# Pause this long (ms) and the recording rolls to a new segment file.
IDLE_MS = 300

# (folder, on-screen instruction)
GESTURES = [
    ("line",   "Draw straight lines / flick back and forth in a line."),
    ("circle", "Move the cursor in smooth circles."),
    ("zigzag", "Sharp zigzags — rapid left/right direction changes."),
    ("free",   "Move freely / aim naturally."),
]
INSTRUCTIONS = dict(GESTURES)


def record_one(gesture, rep, counter, idle_ms=IDLE_MS):
    print(f"\n=== {gesture.upper()}  (rep {rep}) ===")
    print(INSTRUCTIONS[gesture])
    input("Press Enter to START recording...")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(DATA_RAW, gesture, f"{gesture}_{stamp}_{counter}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    log = Logger(path, idle_ms=idle_ms or None)
    log.start_logger()
    hint = (f" (pausing {idle_ms} ms splits a new segment)" if idle_ms else "")
    input(f">> recording...{hint} perform the gesture, "
          "then press Enter to STOP.\n")
    log.stop_logger()

    if not log.segments:
        print("   no movement captured (nothing saved)")
        return

    for seg in log.segments:
        samples = sum(1 for _ in open(seg)) - 1  # minus header
        print(f"   saved {samples:5d} samples -> {os.path.relpath(seg, HERE)}")
    if len(log.segments) > 1:
        print(f"   ({len(log.segments)} segments -- pauses split the take)")


def main():
    print("Guided mouse data collector")
    print("It will tell you a gesture, you do it, press Enter, repeat.")
    print("Ctrl+C to quit.\n")

    choice = tui.pick(
        ["cycle all gestures"] + [g for g, _ in GESTURES],
        "What to collect?",
    )
    if choice is None:
        return
    order = [g for g, _ in GESTURES] if choice == "cycle all gestures" else [choice]

    raw = input("reps per gesture (blank = loop until Ctrl+C): ").strip()
    reps = int(raw) if raw.isdigit() else None

    raw = input(f"idle split in ms (blank = {IDLE_MS}, 0 = off): ").strip()
    idle_ms = int(raw) if raw.isdigit() else IDLE_MS

    counter = itertools.count(1)
    rounds = range(1, reps + 1) if reps else itertools.count(1)
    for rep in rounds:
        for gesture in order:
            record_one(gesture, rep, next(counter), idle_ms)

    print("\nAll requested samples collected.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
