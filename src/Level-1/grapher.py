import matplotlib.pyplot as plot
import pandas as pd

class Plotter:

    def __init__(self, file):
        self.file = file
        self.df = pd.read_csv(file)

    def load_pos(self):
        self.df.plot(x='x', y='y')
        plot.show()

    def load_mag(self):
        self.df.plot(x='Magnitude', y='Direction')
        plot.show()
