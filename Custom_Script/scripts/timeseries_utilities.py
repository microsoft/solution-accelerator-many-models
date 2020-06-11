import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class SimpleLagger(TransformerMixin, BaseEstimator):
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
        assert y is None, \
            "Expected y input to be none. Use target column in input dataframe"

        X_fit = X.copy()
        max_lag_order = max(self.lag_orders)
        self._train_tail = X_fit.sort_index(ascending=True).iloc[-max_lag_order:]

        return self

    def transform(self, X):
        """
        Create lag features of the target for the input data.
        Transform will drop rows with NA values.
        """
        assert list(X.index.names) == [self.time_column_name], \
            "Expected time column to comprise the index of the input Dataframe"
        assert self.target_column_name in X.columns, \
            "Expected target column to be in the input Dataframe"

        X_trans = X.copy()
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

        # Return transformed dataframe with the same time range as X and drop NA rows
        return X_trans.loc[X.index].dropna()


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
        assert y is None, \
            "Expected y input to be none. Use target column in input dataframe"
        X_fit = X.copy()
        y_fit = X_fit.pop(self.target_column_name)
        self._column_order = X_fit.columns
        self.sklearn_model.fit(X_fit.values, y_fit.values)

        return self

    def predict(self, X):
        """
        Predict on the input dataframe.
        Return a Pandas Series with time in the index
        """
        X_pred = X.drop(columns=[self.target_column_name], errors='ignore')[self._column_order]
        y_raw = self.sklearn_model.predict(X_pred.values)
        return pd.Series(data=y_raw, index=X_pred.index)


class Sklearn1StepForecaster:

    def __init__(self, transform_steps, estimator, target_column_name):
        assert estimator is not None, "Estimator cannot be None."
        estimator_step = ('estimator', SklearnWrapper(estimator, target_column_name))
        steps = transform_steps + [estimator_step] if transform_steps is not None else [estimator_step]
        self.pipeline = Pipeline(steps=steps)

    def _forecast_in_sample(self, X):
        pass

    def _recursive_forecast(self, X):
        pass

    def fit(self, X):
        self.pipeline.fit(X)
        return self

    def forecast(self, X):
        pass
