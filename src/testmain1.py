"""Entry point: record a mouse log, or graph an existing one.

Run from the src/ directory:  python testmain1.py
New recordings are written to data/raw/free/; graphing lists every log under
data/raw/ (grouped by gesture folder).
"""
import os
import datetime

from etc import grapher, tui
from logger.mouse_logger import Logger

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(HERE, "..", "data", "raw")

FILTER_CHOICES = ["none", "ma", "ema", "lpf", "kf", "kf2", "lr", "rf",
                  "lrt", "rft", "lrf", "rff", "all"]
_NOPARAM = ("all", "lr", "rf", "lrt", "rft", "lrf", "rff")


if __name__ == "__main__":
    action = tui.pick(["log", "graph"], "What to do?")

    if action == "log":
        name = input("custom name? (blank = timestamp): ").strip()
        if not name:
            name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(DATA_RAW, "free", f"{name}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        l = Logger(path)
        l.start_logger()
        input("logging... press enter to stop\n")
        l.stop_logger()

    elif action == "graph":
        logs = tui.list_logs(DATA_RAW)
        if not logs:
            print("no logs found under data/raw")
        else:
            labels = [label for label, _ in logs]
            chosen = tui.pick(labels, "Pick a log")
            if chosen:
                path = dict(logs)[chosen]
                g = grapher.Plotter(path)
                mode = tui.pick(["position (x/y)", "vector (mag/dir)"], "Plot mode")
                kind = tui.pick(FILTER_CHOICES, "Filter")
                kind = None if kind in (None, "none") else kind
                param = None
                if kind and kind not in _NOPARAM:
                    raw = input(
                        "param (blank = default: ma=window5, ema=alpha0.2, "
                        "lpf=cutoff5Hz, kf=measvar50, kf2=q5): "
                    ).strip()
                    if raw:
                        param = int(raw) if kind == "ma" else float(raw)
                if mode and mode.startswith("position"):
                    g.load_pos(kind, param)
                else:
                    g.load_mag(kind, param)
