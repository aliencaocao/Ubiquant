import numpy as np
import pandas as pd


class TimeSeriesAPI:
    def __init__(self, df):
        df = df.reset_index(drop=True)
        self.df = df
        self.target = df["target"].values

        df_groupby_timeid = df.groupby("time_id")
        self.df_iter = df_groupby_timeid.__iter__()
        self.init_num_timeid = len(df_groupby_timeid)

        self.next_calls = 0
        self.pred_calls = 0

        self.predictions = []
        self.targets = []

    def __iter__(self):
        return self

    def __len__(self):
        return self.init_num_timeid - self.next_calls

    def __next__(self):
        assert self.pred_calls == self.next_calls, "You must call `predict()` before you get the next batch of data."

        time_id, df = next(self.df_iter)
        self.next_calls += 1

        data_df = df  # .drop(columns=["time_id", "target"])

        target_df = df[["row_id", "target", "investment_id"]]
        self.targets.append(target_df)

        pred_df = target_df.drop(columns=["investment_id"])
        pred_df["target"] = 0.

        return data_df, pred_df

    def predict(self, pred_df):
        assert self.pred_calls == self.next_calls - 1, "You must get the next batch before making a new prediction."
        assert pred_df.columns.to_list() == ['row_id', 'target'], "Prediction dataframe have invalid columns."

        pred_df = pred_df.astype({'row_id': np.dtype('str'), 'target': np.dtype('float64')})
        self.predictions.append(pred_df)
        self.pred_calls += 1
