"""This module contains the heads implemented in Discretax."""

from discretax.heads.base import AbstractHead
from discretax.heads.classification import ClassificationHead
from discretax.heads.forecasting import (
    ChannelIndependentTemporalProjectionForecastHead,
    SequenceForecastHead,
    TemporalProjectionForecastHead,
)
from discretax.heads.regression import RegressionHead

__all__ = [
    "AbstractHead",
    "ClassificationHead",
    "SequenceForecastHead",
    "TemporalProjectionForecastHead",
    "ChannelIndependentTemporalProjectionForecastHead",
    "RegressionHead",
]
