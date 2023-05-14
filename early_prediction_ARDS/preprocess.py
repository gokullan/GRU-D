import numpy as np
import pandas as pd

class Preprocess:
    def __init__(self, dataset):
        self.dataset = dataset

    def select_cols(self, cols_to_select):
        self.dataset = self.dataset[cols_to_select]

    def remove_outliers(self, colname):
        Q1 = np.nanpercentile(self.dataset[colname], 20, method = 'midpoint')
        Q3 = np.nanpercentile(self.dataset[colname], 80, method = 'midpoint')
        IQR = Q3 - Q1
        # Upper bound
        upper = Q3 + 1.5 * IQR
        upper_array = np.where(self.dataset[colname]>upper)
        # Lower bound
        lower = Q1 - 1.5 * IQR
        lower_array=np.where(self.dataset[colname]<=lower)
        # Replace the outliers with IQR
        self.dataset.loc[upper_array[0], colname] = IQR
        self.dataset.loc[lower_array[0], colname] = IQR

    def fill_null(self, grouped):
        cols = ['peep', 'pao2fio2ratio']
        for idx in range(len(grouped)):
          _n = len(grouped[idx])
          for col in cols:
            if grouped[idx][col].isna().sum() == _n:
              grouped[idx][col].fillna(0, inplace=True)
            else:
              grouped[idx][col].fillna(method='ffill', inplace=True)
              grouped[idx][col].fillna(method='bfill', inplace=True)

    def group(self):
        samples_grouped = self.dataset.groupby(by='stay_id')
        grouped_and_sorted = []
        for name, group in samples_grouped.__iter__():
          # consider only those records that have at least 10 samples
          if (len(group) > 10):
            grouped_and_sorted.append(group.sort_values(by='charttime').reset_index(drop=True))
        return grouped_and_sorted


