from pynput import mouse
import pandas as pd
import math, os, datetime

class Logger:

    def __init__(self, file):
        self.file = file
        self.listener = mouse.Listener(
            on_move=self.on_move
        )
        self.first_time = True
        self.prevx = 0
        self.prevy = 0

    def on_move(self, x, y):
        print(f"Pointer moved to ({x},{y})")
        if not self.first_time:
            xs = (x-self.prevx)**2
            ys = (y-self.prevy)**2
            print(f"Magnitude: {math.sqrt(xs+ys)} | Direction: {math.atan2((y-self.prevy), (x-self.prevx))}")
            data = pd.DataFrame({
                'x': [x],
                'y': [y],
                'Magnitude': [math.sqrt(xs+ys)],
                'Direction': [math.atan2((y-self.prevy), (x-self.prevx))],
                'time': [str(datetime.datetime.now())]
            })
            if os.path.exists(self.file):
                data.to_csv(self.file, mode='a', header=False, index=False)
            else:
                data.to_csv(self.file, mode='w', header=True, index=False)

        else:
            self.first_time = False
        self.prevx = x
        self.prevy = y

    def start_logger(self):
        self.listener.start()

    def stop_logger(self):
        self.listener.stop()
