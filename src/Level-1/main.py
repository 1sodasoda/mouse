import grapher, logger, tui, datetime

if __name__ == "__main__":
    action = tui.pick(["log", "graph"], "What to do?")

    if action == "log":
        name = input("custom name? (blank = timestamp): ").strip()
        if not name:
            name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        l = logger.Logger(f"./logs/{name}.csv")
        l.start_logger()
        input("logging... press enter to stop\n")
        l.stop_logger()

    elif action == "graph":
        logs = tui.list_logs("./logs")
        if not logs:
            print("no logs found in ./logs")
        else:
            chosen = tui.pick(logs, "Pick a log")
            if chosen:
                g = grapher.Plotter(f"./logs/{chosen}.csv")
                mode = tui.pick(["position (x/y)", "vector (mag/dir)"], "Plot mode")
                kind = tui.pick(["none", "ma", "ema", "lpf"], "Filter")
                kind = None if kind in (None, "none") else kind
                param = None
                if kind:
                    raw = input(
                        "param (blank = default: ma=window5, ema=alpha0.2, lpf=cutoff5Hz): "
                    ).strip()
                    if raw:
                        param = int(raw) if kind == "ma" else float(raw)
                if mode and mode.startswith("position"):
                    g.load_pos(kind, param)
                else:
                    g.load_mag(kind, param)
