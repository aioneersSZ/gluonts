import docutils
from typing import Optional

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.util import forecast_start
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.predictor import RepresentablePredictor

from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ExponentialSmoothingPredictor(RepresentablePredictor):
    """ExponentialSmoothingPredictor
    Wrapper for calling the statsmodels.tsa.holtwinters.ExponentialSmoothing package.
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html

    Parameters
    ----------
    prediction_length : int
        Number of time steps to be predicted.
    freq : Optional[str], optional
        The granularity of the time series (e.g. '1H'), by default None
    target_dtype : Optional[type], optional
        Data type of the target values and forecast values, by default np.float32
    replace_neg_values : Optional[bool], optional
        Replace negative values by 0, by default True
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        freq: Optional[str] = None,
        target_dtype: Optional[type] = np.float32,
        replace_neg_values: Optional[bool] = True,
        **kwargs
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        self.freq = freq
        self.prediction_length = prediction_length
        self.target_dtype = target_dtype
        self.replace_neg_values = replace_neg_values

        self.__dict__.update(**kwargs)
        self.kwargs = kwargs

    def predict_item(self, item: DataEntry, num_samples: int = 1, **kwargs) -> Forecast:
        """predict_item
        Make time series prediction with the "fit" method.
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.fit.html#statsmodels.tsa.holtwinters.ExponentialSmoothing.fit

        Parameters
        ----------
        item : DataEntry
            The item of the dataset.
        num_samples : int, optional
            Num of samples to be generated, by default 1.

        Returns
        -------
        Forecast
            Predictions for one item.
        """

        endog = list(map(self.target_dtype, item["target"]))

        num_periods = len(item.get("target"))
        forecast_start_time = forecast_start(item)
        dates = pd.date_range(
            start=forecast_start_time.to_timestamp(),
            freq=self.freq,
            periods=num_periods,
        )

        assert len(endog) >= 1, "all time series should have at least one data point"

        ts_predictor = ExponentialSmoothing(
            endog=endog,
            freq=self.freq,
            dates=dates,
            **docutils.filter_kwargs(ExponentialSmoothing.__init__, self.kwargs)
        ).fit(**docutils.filter_kwargs(ExponentialSmoothing.fit, self.kwargs))

        prediction = ts_predictor.forecast(steps=self.prediction_length)
        prediction = list(map(self.target_dtype, prediction))
        samples = np.array([prediction])

        if self.replace_neg_values:
            samples = np.where(samples < 0, 0, samples)

        return SampleForecast(
            samples=samples,
            start_date=forecast_start_time,
            item_id=item.get("item_id", None),
        )
