"""
Data preprocessing utilities for digital twin sensor streams.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


class SensorDataNormalizer:
    """
    Fit and apply min-max or z-score normalisation to sensor data.

    Parameters
    ----------
    method : 'zscore' (default) or 'minmax'
    """

    def __init__(self, method: str = "zscore") -> None:
        if method not in ("zscore", "minmax"):
            raise ValueError("method must be 'zscore' or 'minmax'")
        self.method = method
        self._loc: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> "SensorDataNormalizer":
        """Compute normalisation parameters from training data."""
        data = np.asarray(data, dtype=float)
        if self.method == "zscore":
            self._loc = data.mean(axis=0)
            self._scale = data.std(axis=0)
            self._scale[self._scale < 1e-12] = 1.0  # avoid division by zero
        else:
            self._loc = data.min(axis=0)
            self._scale = data.max(axis=0) - data.min(axis=0)
            self._scale[self._scale < 1e-12] = 1.0
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (np.asarray(data, dtype=float) - self._loc) / self._scale

    def inverse_transform(self, data_norm: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(data_norm, dtype=float) * self._scale + self._loc

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")


class FeatureExtractor:
    """
    Extract statistical features from fixed-length time-series windows.
    Used to convert raw sensor segments into feature vectors for anomaly detection.
    """

    @staticmethod
    def extract(segment: np.ndarray) -> dict:
        """
        Compute standard time-domain features from a 1-D signal segment.

        Parameters
        ----------
        segment : 1-D array of sensor values

        Returns
        -------
        dict of feature name → scalar value
        """
        from scipy.stats import kurtosis, skew

        x = np.asarray(segment, dtype=float)
        n = len(x)
        if n < 2:
            return {}

        # Gradient trend
        trend = np.polyfit(np.arange(n), x, 1)[0]

        return {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)),
            "rms": float(np.sqrt((x**2).mean())),
            "peak": float(x.max()),
            "trough": float(x.min()),
            "peak_to_peak": float(x.max() - x.min()),
            "kurtosis": float(kurtosis(x, fisher=True)),
            "skewness": float(skew(x)),
            "trend_slope": float(trend),
            "energy": float((x**2).sum()),
            "crest_factor": float(x.max() / (np.sqrt((x**2).mean()) + 1e-12)),
        }

    @staticmethod
    def batch_extract(segments: np.ndarray) -> pd.DataFrame:
        """
        Extract features from multiple segments.

        Parameters
        ----------
        segments : (n_segments, segment_length)

        Returns
        -------
        pd.DataFrame with one row per segment
        """
        return pd.DataFrame([
            FeatureExtractor.extract(seg) for seg in segments
        ])


class DataLogger:
    """
    Append timestamped state entries to a pandas DataFrame and save to CSV.
    """

    def __init__(self) -> None:
        self._records: list[dict] = []
        self._df: Optional[pd.DataFrame] = None

    def append(self, timestamp: float, data: dict) -> None:
        record = {"timestamp": timestamp}
        record.update({k: v for k, v in data.items() if not isinstance(v, (list, dict))})
        self._records.append(record)
        self._df = None  # invalidate cache

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.DataFrame(self._records)
        return self._df

    def save_csv(self, path: str) -> None:
        self.dataframe.to_csv(path, index=False)

    def load_csv(self, path: str) -> None:
        self._df = pd.read_csv(path)
        self._records = self._df.to_dict("records")

    def __len__(self) -> int:
        return len(self._records)
