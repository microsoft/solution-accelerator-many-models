import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class SimpleLagger(TransformerMixin, BaseEstimator):
    """
    Simple lagging transform that creates lagged values of the target column.
    This transform uses information known at fit time to create lags at transform time
    to maintain lag feature continuity across train/test splits
    """

    def __init__(self, target_column_name, lag_orders=[1]):
        self.target_column_name = target_column_name
        self.lag_orders = lag_orders

    def fit(self, X, y=None):
        """
        Fit the lagger transform.
        This transform caches the tail of the training data up to the maximum lag order
        so that lag features can be created on test data.
        """
        assert y is None, \
            "Expected y input to be none. Use target column in input dataframe"

        X_fit = X.copy()
        max_lag_order = max(self.lag_orders)
        self._train_tail = X_fit.sort_index(ascending=True).iloc[-max_lag_order:]
        self._column_order = self._train_tail.columns

        return self

    def transform(self, X):
        """
        Create lag features of the target for the input data.
        Transform will drop rows with NA values.
        """
        X_trans = X.copy()
        added_target = False
        if self.target_column_name not in X_trans.columns:
            X_trans[self.target_column_name] = np.nan
            added_target = True

        X_trans.sort_index(ascending=True, inplace=True)

        # decide if we need to use the training cache i.e. are we in a test scenario?
        train_latest = self._train_tail.index.max()
        X_earliest = X_trans.index.min()
        if train_latest < X_earliest:
            # X data is later than the training period - append the cached tail of training data
            X_trans = pd.concat((self._train_tail, X_trans[self._column_order]))

        # Make the lag features
        for lag_order in self.lag_orders:
            X_trans['lag_' + str(lag_order)] = X_trans[self.target_column_name].shift(lag_order)

        # Return transformed dataframe with the same time range as X
        if added_target:
            X_trans.drop(columns=[self.target_column_name], inplace=True)
        return X_trans.loc[X.index]


class SklearnWrapper(BaseEstimator):
    """
    Wrapper class around an sklearn model
    """
    def __init__(self, sklearn_model, target_column_name):
        self.sklearn_model = sklearn_model
        self.target_column_name = target_column_name

    def fit(self, X, y=None):
        """
        Fit the sklearn model on the input dataframe.
        """
        assert self.target_column_name in X.columns, \
            "Target column is missing from the input dataframe."
        X_fit = X.copy().dropna()
        assert len(X_fit) > 0, 'Training dataframe is empty after dropping NA values'
        y_fit = X_fit.pop(self.target_column_name)
        self._column_order = X_fit.columns
        self.sklearn_model.fit(X_fit.values, y_fit.values)
        return self

    def predict(self, X):
        """
        Predict on the input dataframe.
        Return a Pandas Series with time in the index
        """
        X_pred = X.drop(columns=[self.target_column_name], errors='ignore')[self._column_order].dropna()
        assert len(X_pred) > 0, 'Prediction dataframe is empty after dropping NA values'
        y_raw = self.sklearn_model.predict(X_pred.values)
        return pd.Series(data=y_raw, index=X_pred.index)


class SimpleForecaster:
    """
    Forecasting class for a simple, 1-step ahead forecaster.
    This class encapsulates fitting a transform pipeline with an sklearn regression estimator
    and producing in-sample and out-of-sample forecasts.
    Out-of-sample forecasts apply the model recursively over the test set to produce predictions
    at any forecast horizon.

    The forecaster assumes that the time-series data is regularly sampled on a contiguous interval;
    it does not handle missing values.
    """

    def __init__(self, transform_steps, estimator, target_column_name, time_column_name):
        assert estimator is not None, "Estimator cannot be None."
        assert transform_steps is None or isinstance(transform_steps, list), \
            "transform_steps should be a list"
        estimator_step = ('estimator', SklearnWrapper(estimator, target_column_name))
        steps = transform_steps + [estimator_step] if transform_steps is not None else [estimator_step]
        self.pipeline = Pipeline(steps=steps)

        self.target_column_name = target_column_name
        self.time_column_name = time_column_name

    def _recursive_forecast(self, X):
        """
        Apply the trained model resursively for out-of-sample predictions.
        """
        X_fcst = X.copy().sort_index(ascending=True)
        if self.target_column_name not in X_fcst.columns:
            X_fcst[self.target_column_name] = np.nan

        forecasts = pd.Series(np.nan, index=X_fcst.index)
        for fcst_date in X_fcst.index.get_level_values(self.time_column_name):
            # Get predictions on an expanding window ending on the current forecast date
            y_fcst = self.pipeline.predict(X_fcst[X_fcst.index <= fcst_date])

            # Set the current forecast
            forecasts.loc[fcst_date] = y_fcst.loc[fcst_date]

            # Set the actual value to the forecast value so that lag features can be made on next iteration
            X_fcst.loc[fcst_date, self.target_column_name] = y_fcst.loc[fcst_date]

        return forecasts

    def fit(self, X):
        """
        Fit the forecasting pipeline.
        This method assumes the target is a column in the input, X.
        """
        assert list(X.index.names) == [self.time_column_name], \
            "Expected time column to comprise input dataframe index."
        self._latest_training_date = X.index.max()
        self.pipeline.fit(X)
        return self

    def forecast(self, X):
        """
        Make forecasts over the prediction frame, X.
        X can contain in-sample and out-of-sample data.
        For out-of-sample data, the 1-step-ahead model is recursively applied.

        Returns forecasts for the target in a pd.Series object with the same time index as X.
        np.nan values will be returned for dates where a forecast could not be found.  
        """
        assert list(X.index.names) == [self.time_column_name], \
            "Expected time column to comprise input dataframe index."
        # Get in-sample forecasts if requested
        X_insamp = X[X.index <= self._latest_training_date]
        forecasts_insamp = pd.Series()
        if len(X_insamp) > 0:
            forecasts_insamp = self.pipeline.predict(X_insamp)

        # Get out-of-sample forecasts
        X_fcst = X[X.index > self._latest_training_date]
        forecasts = pd.Series()
        if len(X_fcst) > 0:
            # Need to iterate/recurse 1-step forecasts here
            forecasts = self._recursive_forecast(X_fcst)
        forecasts = pd.concat((forecasts_insamp, forecasts))

        return forecasts.reindex(X.index)
