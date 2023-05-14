import math
import numpy as np

class SampleMaker:
    def __init__(self, grouped):
        self.grouped = grouped
        for idx in range(len(self.grouped)):
            self.grouped[idx] = self.grouped[idx].set_index('charttime')
            self.grouped[idx].drop(columns=['hadm_id', 'stay_id'], inplace=True)

    @staticmethod
    def find_onset_time(stay):
        idx = 0
        for index, row in stay.iterrows():
          if row['peep'] >= 5 and row['pao2fio2ratio'] < 300:
            return idx
          idx += 1
        return -1

    @staticmethod
    # returns the (integer) hours difference between the 2 timestamps
    def hours_diff(t1, t2):
        diff = math.floor(
            (t2 - t1).total_seconds() / 3600
        )
        return diff

    @staticmethod
    def append(np_arr, to_append):
        np_arr = np.append(np_arr, np.empty(shape=(1,), dtype=object), axis=0)
        np_arr[-1] = to_append
        return np_arr

    def make_samples(self, pred_window=12, bound=72, min_size=3):
        positives = np.empty(shape=(0,), dtype=object)
        negatives = np.empty(shape=(0,), dtype=object)
        pos_timestamps = np.empty(shape=(0,), dtype=object)
        neg_timestamps = np.empty(shape=(0,), dtype=object)
        for group in self.grouped:
          n = len(group)
          onset_time_index = SampleMaker.find_onset_time(group)
          if onset_time_index != -1:
            onset_time = group.index[onset_time_index]
            # number of hours before the onset time for which data is available
            n_hours = SampleMaker.hours_diff(group.index[0], group.index[onset_time_index])
            if n_hours < pred_window:  # insufficient data
              continue
            else:
              # find the first `charttime` for which the difference with `onset_time`
              # is less than `bound`
              start_idx = -1
              for i in range(onset_time_index):
                if SampleMaker.hours_diff(group.index[i], group.index[onset_time_index]) <= bound:
                  start_idx = i
                  break
              if start_idx != -1 and (onset_time_index - start_idx + 1) >= min_size:
                positives = SampleMaker.append(positives, group.iloc[start_idx:onset_time_index].to_numpy())
                pos_timestamps = SampleMaker.append(pos_timestamps, group.index[start_idx:onset_time_index].to_numpy())
          else:
            if n >= min_size:
              negatives = SampleMaker.append(negatives, group.to_numpy())
              neg_timestamps = SampleMaker.append(neg_timestamps, group.index.to_numpy())
        return positives, negatives, pos_timestamps, neg_timestamps

    def make_rnn_samples(self, samples, lookback):
        rnn_samples = np.empty(shape=(0,), dtype=object)
        for sample in samples:
          n = sample.shape[0]
          for i in range(n - lookback):
            rnn_samples = SampleMaker.append(rnn_samples, sample[i: i + lookback])
        return rnn_samples
    
