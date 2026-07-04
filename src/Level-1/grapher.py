import matplotlib.pyplot as plot
import pandas as pd

class plotter:

    def __init__(self, file):
        self.file = file
        df = pd.read_csv(file)

    def load_pos(self):
        df.plot(x='x', y='y')
        plot.show()
