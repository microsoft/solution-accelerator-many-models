import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class SimpleLagger(BaseEstimator):
    """
    Simple lagging transform that creates lagged values of the target column.
    This transform uses information known at fit time to create lags at transform time
    to maintain lag feature continuity across train/test splits
    """

    def __init__(self, target_column_name, time_column_name, lag_orders=[1]):
        self.target_column_name = target_column_name
        self.time_column_name = time_column_name
        self.lag_orders = lag_orders

    def fit(self, X, y=None):
        """
        Fit the lagger transform.
        This transform caches the tail of the training data up to the maximum lag order
        so that lag features can be created on test data.
        """
        assert list(X.index.names) == [self.time_column_name], \
            "Expected time column to comprise the index of the input Dataframe"

        X_fit = X.copy()
        if y is not None:
            X_fit[self.target_column_name] = y
        else:
            X_fit[self.target_column_name] = np.nan

        max_lag_order = max(self.lag_orders)
        self._train_tail = X_fit.sort_index(ascending=True).iloc[-max_lag_order:]

        return self

    def transform(self, X, y=None):
        """
        Create lag features of the target for the input data.
        """
        assert list(X.index.names) == [self.time_column_name], \
            "Expected time column to comprise the index of the input Dataframe"

        X_trans = X.copy()
        if y is not None:
            X_trans[self.target_column_name] = y
        else:
            X_trans[self.target_column_name] = np.nan
        X_trans.sort_index(ascending=True, inplace=True)

        # decide if we need to use the training cache i.e. are we in a test scenario?
        train_latest = self._train_tail.index.max()
        X_earliest = X_trans.index.min()
        if train_latest < X_earliest:
            # X data is later than the training period - append the cached tail of training data
            X_trans = pd.concat((self._train_tail, X_trans))

        # Make the lag features
        for lag_order in self.lag_orders:
            X_trans['lag_' + str(lag_order)] = X_trans[self.target_column_name].shift(lag_order)

        # Extract the original time window, drop the target, and remove rows with NA values
        return X_trans[X_trans.index >= X_earliest].drop(columns=[self.target_column_name]).dropna()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
