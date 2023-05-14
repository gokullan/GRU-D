import numpy as np
import pandas as pd

class Refactor:
    def __init__(self, inputs, labels, timestamps):
        self.n = len(inputs)
        self.lookback = len(timestamps[0])
        self.inputs = inputs
        self.labels = labels
        self.timestamps = timestamps

    @staticmethod
    def mins_diff(t1, t2):
        diff = (t2 - t1).astype('timedelta64[m]')
        return diff

    @staticmethod
    def generate_timestamps_min(timestamps_i):
        timestamps_i_new = []
        timestamps_i_new.append(0)
        for i in range(1, len(timestamps_i)):
          timestamps_i_new.append(Refactor.mins_diff(timestamps_i[0], timestamps_i[i]))
        return np.array(timestamps_i_new, dtype=np.float32)

    def standardize_timestamps(self):
        timestamps_new = np.empty(shape=(0,), dtype=object)
        for i in range(self.n):
            timestamps_new = np.append(timestamps_new, np.empty(shape=(1,), dtype=object), axis=0)
            timestamps_new[-1] = Refactor.generate_timestamps_min(self.timestamps[i])
        return timestamps_new

    def get_masks(self):
        masks = np.empty(shape=(self.n,), dtype=object)
        for i in range(self.n):
            masks[i] = np.logical_not(pd.isna(self.inputs[i])).astype(int)
        return masks
