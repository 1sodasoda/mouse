import curses
import os
import sys


def list_logs(logs_dir="./logs"):
    """Return log file basenames (newest first), no .csv extension."""
    if not os.path.isdir(logs_dir):
        return []
    files = [f for f in os.listdir(logs_dir) if f.endswith(".csv")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(logs_dir, f)),
               reverse=True)
    return [f[:-4] for f in files]


def pick(options, title="Select"):
    """Arrow-key menu. Returns the chosen option, or None if cancelled.

    Falls back to a numbered prompt when no interactive terminal is present.
    """
    if not options:
        return None
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return _pick_numbered(options, title)
    try:
        return curses.wrapper(_menu, options, title)
    except curses.error:
        return _pick_numbered(options, title)


def _pick_numbered(options, title):
    print(title)
    for i, o in enumerate(options):
        print(f"  {i + 1}) {o}")
    raw = input("choice #: ").strip()
    if not raw.isdigit():
        return None
    i = int(raw) - 1
    return options[i] if 0 <= i < len(options) else None


def _menu(stdscr, options, title):
    curses.curs_set(0)
    idx = 0
    top = 0  # first visible row (scroll offset)
    while True:
        stdscr.clear()
        height, _ = stdscr.getmaxyx()
        body = max(1, height - 3)
        if idx < top:
            top = idx
        elif idx >= top + body:
            top = idx - body + 1

        _put(stdscr, 0, 0, title, curses.A_BOLD)
        _put(stdscr, 1, 0, "↑/↓ move · enter select · q cancel")
        for row, i in enumerate(range(top, min(top + body, len(options)))):
            marker = "> " if i == idx else "  "
            attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
            _put(stdscr, row + 3, 0, marker + str(options[i]), attr)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (curses.KEY_UP, ord("k")):
            idx = (idx - 1) % len(options)
        elif k in (curses.KEY_DOWN, ord("j")):
            idx = (idx + 1) % len(options)
        elif k in (curses.KEY_ENTER, 10, 13):
            return options[idx]
        elif k in (27, ord("q")):
            return None


def _put(stdscr, y, x, text, attr=curses.A_NORMAL):
    """addstr that won't crash on narrow/short terminals."""
    height, width = stdscr.getmaxyx()
    if y >= height:
        return
    try:
        stdscr.addstr(y, x, text[: max(0, width - x - 1)], attr)
    except curses.error:
        pass
