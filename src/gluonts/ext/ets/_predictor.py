from typing import Optional, Union, Iterator, Dict

import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.dataset import Dataset
from gluonts.dataset.common import DataEntry
from gluonts.dataset.util import forecast_start
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.predictor import RepresentablePredictor

from statsmodels.tsa.holtwinters import ExponentialSmoothing


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
    trend : {"add", "mul", "additive", "multiplicative", None}, optional
        Type of trend component.
    damped_trend : bool, optional
        Should the trend component be damped.
    seasonal : {"add", "mul", "additive", "multiplicative", None}, optional
        Type of seasonal component.
    seasonal_periods : int, optional
        The number of periods in a complete seasonal cycle, e.g., 4 for
        quarterly data or 7 for daily data with a weekly cycle.
    initialization_method : str, optional
        Method for initialize the recursions. One of:

        * None
        * 'estimated'
        * 'heuristic'
        * 'legacy-heuristic'
        * 'known'

        None defaults to the pre-0.12 behavior where initial values
        are passed as part of ``fit``. If any of the other values are
        passed, then the initial values must also be set when constructing
        the model. If 'known' initialization is used, then `initial_level`
        must be passed, as well as `initial_trend` and `initial_seasonal` if
        applicable. Default is 'estimated'. "legacy-heuristic" uses the same
        values that were used in statsmodels 0.11 and earlier.
    initial_level : float, optional
        The initial level component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    initial_trend : float, optional
        The initial trend component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    initial_seasonal : array_like, optional
        The initial seasonal component. An array of length `seasonal`
        or length `seasonal - 1` (in which case the last initial value
        is computed to make the average effect zero). Only used if
        initialization is 'known'. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    use_boxcox : {True, False, 'log', float}, optional
        Should the Box-Cox transform be applied to the data first? If 'log'
        then apply the log. If float then use the value as lambda.
    bounds : dict[str, tuple[float, float]], optional
        An dictionary containing bounds for the parameters in the model,
        excluding the initial values if estimated. The keys of the dictionary
        are the variable names, e.g., smoothing_level or initial_slope.
        The initial seasonal variables are labeled initial_seasonal.<j>
        for j=0,...,m-1 where m is the number of period in a full season.
        Use None to indicate a non-binding constraint, e.g., (0, None)
        constrains a parameter to be non-negative.
    dates : array_like of datetime, optional
        An array-like object of datetime objects. If a Pandas object is given
        for endog, it is assumed to have a DateIndex.
    freq : str, optional
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
        'M', 'A', or 'Q'. This is optional if dates are given.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        freq: Optional[str] = None,
        target_dtype: Optional[type] = np.float32,
        replace_neg_values: Optional[bool] = True,
        trend: Optional[str] = None,
        damped_trend: Optional[bool] = False,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        initialization_method: Optional[str] = "estimated",
        initial_level: float = None,
        initial_trend: float = None,
        initial_seasonal: Optional[str] = None,
        use_boxcox: Union[str, bool, float] = False,
        bounds: dict = None,
        missing: str = "none",
        **kwargs,
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        self.freq = freq
        self.prediction_length = prediction_length
        self.target_dtype = target_dtype
        self.replace_neg_values = replace_neg_values
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.initialization_method = initialization_method
        self.initial_level = initial_level
        self.initial_trend = initial_trend
        self.initial_seasonal = initial_seasonal
        self.use_boxcox = use_boxcox
        self.bounds = bounds
        self.missing = missing

        self.kwargs = kwargs

    def predict(
        self, dataset: Dataset, num_samples: Optional[int]
    ) -> Iterator[Forecast]:
        for item in dataset:
            yield self.predict_item(item=item, num_samples=num_samples)

    def predict_item(
        self, item: DataEntry, num_samples: Optional[int]
    ) -> Forecast:
        """predict_item
        Make time series prediction with the "fit" method.
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.fit.html#statsmodels.tsa.holtwinters.ExponentialSmoothing.fit

        Parameters
        ----------
        item : DataEntry
            The item of the dataset.
        num_samples : Optional[int]
            The number of sample paths, which are all the same.

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

        assert (
            len(endog) >= 1
        ), "all time series should have at least one data point"

        item_estimator = ExponentialSmoothing(
            endog=endog,
            freq=self.freq,
            dates=dates,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method=self.initialization_method,
        )

        # Get a predictor from the estimator by fitting on the data
        item_predictor = item_estimator.fit(**self.kwargs)

        prediction = item_predictor.forecast(steps=self.prediction_length)
        prediction = list(map(self.target_dtype, prediction))
        prediction = np.array(prediction)

        if self.replace_neg_values:
            prediction = np.where(prediction < 0, 0, prediction).tolist()

        samples = [prediction for _ in range(num_samples)]
        samples = np.array(samples)

        return SampleForecast(
            samples=samples,
            start_date=forecast_start_time,
            item_id=item.get("item_id", None),
        )
