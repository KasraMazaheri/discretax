"""This module contains the encoders implemented in Discretax."""

from discretax.encoder.base import AbstractEncoder
from discretax.encoder.embedding import EmbeddingEncoder
from discretax.encoder.image_patch import ImagePatchEncoder
from discretax.encoder.linear import LinearEncoder
from discretax.encoder.time_series_patch import TimeSeriesPatchEncoder

__all__ = [
    "AbstractEncoder",
    "LinearEncoder",
    "EmbeddingEncoder",
    "ImagePatchEncoder",
    "TimeSeriesPatchEncoder",
]
