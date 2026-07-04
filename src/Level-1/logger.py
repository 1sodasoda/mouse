from pynput import mouse
import pandas as pd
import math

class Logger:

    def __init__(self, file):
        self.file = file
        self.listener = mouse.Listener(
            on_move=on_move
        )
        self.first_time = true
        self.prevx = 0
        self.prevy = 0

    def on_move(self, x, y):
        print(f"Pointer moved to ({x},{y})")
        if not self.first_time:
            print(f"Magnitude: {math.sqrt((x-self.prevx)^2+(y-self.prevy)^2)} | Direction: {math.atan2((y-self.prevy), (x-self.prevx))}")
        else:
            self.first_time = false;
        self.prevx = x
        self.prevy = y

    def start_logger(self):
        self.listener.start()

    def stop_logger(self):
        self.listener.stop()
