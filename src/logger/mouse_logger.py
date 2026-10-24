"""Live mouse-movement recorder.

Listens to pointer motion via pynput and appends each move to a CSV with
position, velocity (magnitude/direction), and a force-direction discrepancy.
The read/analysis side of logged data lives in ``commons/mouse_logger.py``.

Recording is quiet by default: at the ~1000 Hz the pointer reports, printing
every move floods the terminal and buries the caller's own prompts. Pass
``verbose=True`` to echo each sample while debugging.

``idle_ms`` splits a recording at pauses. pynput only fires while the pointer
moves, so holding still writes nothing and leaves a time gap in the CSV --
and the row written when motion resumes holds a delta measured *across* the
pause, which is a bogus velocity. With ``idle_ms`` set, a gap that long ends
the current segment and the next movement starts a fresh file, so every saved
file is one stretch of continuous motion with no gap and no cross-pause delta.
Training keeps more of the data that way: ``ml/windowing.py`` drops windows
spanning a pause longer than its own threshold.
"""
from pynput import mouse
import pandas as pd
import math, os, datetime, time


class Logger:

    def __init__(self, file, verbose=False, idle_ms=None):
        self.file = file
        self.base = file
        self.verbose = verbose
        self.idle_ms = idle_ms
        self.segments = []          # files actually written, in order
        self.segment = 1
        self.last_move = None       # perf_counter of the previous move
        self.listener = mouse.Listener(
            on_move=self.on_move
        )
        self.first_time = True
        self.prevx = 0
        self.prevy = 0
        self.prevmag = 0
        self.prevdir = 0

    def _segment_path(self, n):
        """Path for segment n. Segment 1 keeps the caller's filename."""
        if n == 1:
            return self.base
        stem, ext = os.path.splitext(self.base)
        return f"{stem}_part{n}{ext}"

    def _check_idle(self):
        """Roll to a new segment when the pointer has been still too long.

        Called on each move, so the pause is detected by the movement that
        ends it -- no timer thread needed. Resets the delta state as well, so
        the first row of a segment never carries a velocity measured across
        the pause.
        """
        now = time.perf_counter()
        if (self.idle_ms and self.last_move is not None
                and (now - self.last_move) * 1000.0 > self.idle_ms):
            self.segment += 1
            self.file = self._segment_path(self.segment)
            self.first_time = True
            if self.verbose:
                print(f"idle > {self.idle_ms} ms -- new segment "
                      f"{os.path.basename(self.file)}")
        self.last_move = now

    def on_move(self, x, y):
        if self.verbose:
            print(f"Pointer moved to ({x},{y})")
        self._check_idle()
        if not self.first_time:
            xs = (x-self.prevx)**2
            ys = (y-self.prevy)**2
            if self.verbose:
                print(f"Magnitude: {math.sqrt(xs+ys)} | Direction: {math.atan2((y-self.prevy), (x-self.prevx))}")
            # Vector stuff
            prevxcomp = self.prevmag*math.cos(self.prevdir)
            prevycomp = self.prevmag*math.sin(self.prevdir)
            curxcomp = math.sqrt(xs+ys)*math.cos(math.atan2((y-self.prevy), (x-self.prevx)))
            curycomp = math.sqrt(xs+ys)*math.sin(math.atan2((y-self.prevy), (x-self.prevx)))
            chxcomp = curxcomp - prevxcomp
            chycomp = curycomp - prevycomp
            forcedir = math.atan2(chycomp, chxcomp)
            diff = math.atan2((y-self.prevy), (x-self.prevx))-forcedir

            data = pd.DataFrame({
                'x': [x],
                'y': [y],
                'Magnitude': [math.sqrt(xs+ys)],
                'Direction': [math.atan2((y-self.prevy), (x-self.prevx))], #Direction based on velocity
                'Force_Direction': [forcedir],
                'Direction_Desync': [abs(diff)/(math.pi)], # 1 is full desync, 0 = same direction
                # explicit format: str(datetime) drops ".000000" when the
                # microseconds are exactly zero, which breaks readers that
                # infer one timestamp format for the whole file.
                'time': [datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S.%f')]
            })
            if os.path.exists(self.file):
                data.to_csv(self.file, mode='a', header=False, index=False)
            else:
                data.to_csv(self.file, mode='w', header=True, index=False)
                self.segments.append(self.file)
            self.prevmag = math.sqrt(xs+ys)
            self.prevdir = math.atan2((y-self.prevy), (x-self.prevx))
        else:
            self.first_time = False
        self.prevx = x
        self.prevy = y

    def start_logger(self):
        self.listener.start()

    def stop_logger(self):
        self.listener.stop()
