"""
Digital twin state monitor — maintains current system state and detects anomalies.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnomalyEvent:
    timestamp: float
    channel: str
    value: float
    score: float
    threshold: float


class SensorDataProcessor:
    """
    Rolling-window statistics and outlier detection for raw sensor streams.

    Parameters
    ----------
    window : int — rolling window size (number of samples)
    """

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._buffers: dict[str, deque] = {}

    def update(self, channel: str, value: float) -> dict:
        """Add a new reading and return current rolling statistics."""
        if channel not in self._buffers:
            self._buffers[channel] = deque(maxlen=self.window)
        buf = self._buffers[channel]
        buf.append(value)
        arr = np.array(buf)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return {
            "channel": channel,
            "value": value,
            "rolling_mean": float(arr.mean()),
            "rolling_std": float(arr.std()),
            "rolling_rms": float(np.sqrt((arr**2).mean())),
            "peak_to_peak": float(arr.max() - arr.min()),
            "is_outlier": bool(value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr),
        }

    def get_features(self, channel: str) -> dict:
        """Extract statistical features from the rolling buffer."""
        if channel not in self._buffers or len(self._buffers[channel]) < 2:
            return {}
        arr = np.array(self._buffers[channel])
        from scipy.stats import kurtosis, skew
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "rms": float(np.sqrt((arr**2).mean())),
            "peak_to_peak": float(arr.max() - arr.min()),
            "kurtosis": float(kurtosis(arr)),
            "skewness": float(skew(arr)),
            "trend": float(np.polyfit(np.arange(len(arr)), arr, 1)[0]),  # slope
        }

    def reset(self, channel: Optional[str] = None) -> None:
        if channel:
            self._buffers.pop(channel, None)
        else:
            self._buffers.clear()


class DigitalTwinStateMonitor:
    """
    Maintain and update the current state of a physical asset's digital twin.

    State is updated by blending sensor readings with ROM predictions.
    Anomaly detection uses Mahalanobis distance from the nominal operating envelope.

    Parameters
    ----------
    state_variables   : list of variable names to track
    anomaly_threshold : Mahalanobis distance above which an anomaly is flagged
    """

    def __init__(
        self,
        state_variables: list[str],
        anomaly_threshold: float = 3.0,
    ) -> None:
        self.state_vars = state_variables
        self.anomaly_threshold = anomaly_threshold
        self._state: dict[str, float] = {v: 0.0 for v in state_variables}
        self._history: list[dict] = []
        self._nominal_mean: Optional[np.ndarray] = None
        self._nominal_cov_inv: Optional[np.ndarray] = None
        self._anomaly_log: list[AnomalyEvent] = []

    def calibrate_nominal(self, nominal_data: np.ndarray) -> None:
        """
        Fit the nominal operating envelope from historical data.

        Parameters
        ----------
        nominal_data : (n_samples, n_state_vars) — normal operating data
        """
        self._nominal_mean = nominal_data.mean(axis=0)
        cov = np.cov(nominal_data.T)
        # Add small regularisation for numerical stability
        cov += 1e-6 * np.eye(cov.shape[0])
        self._nominal_cov_inv = np.linalg.inv(cov)

    def update_state(
        self,
        timestamp: float,
        sensor_readings: dict[str, float],
        rom_predictions: Optional[dict[str, float]] = None,
        sensor_weight: float = 0.7,
    ) -> dict:
        """
        Fuse sensor readings and ROM predictions to update state.

        Parameters
        ----------
        timestamp      : simulation / wall-clock time
        sensor_readings: dict of measured values
        rom_predictions: dict of ROM-predicted values (may be None)
        sensor_weight  : weight for sensor vs ROM (0–1); 1 = pure sensor

        Returns
        -------
        dict with updated state and anomaly flag
        """
        for var in self.state_vars:
            s = sensor_readings.get(var, None)
            r = rom_predictions.get(var, None) if rom_predictions else None

            if s is not None and r is not None:
                self._state[var] = sensor_weight * s + (1 - sensor_weight) * r
            elif s is not None:
                self._state[var] = s
            elif r is not None:
                self._state[var] = r

        anomaly_score = self._compute_anomaly_score()
        is_anomaly = anomaly_score > self.anomaly_threshold

        entry = {
            "timestamp": timestamp,
            **{f"state_{k}": v for k, v in self._state.items()},
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
        }
        self._history.append(entry)

        if is_anomaly:
            # Log the variable with highest deviation
            worst_var = max(
                self.state_vars,
                key=lambda v: abs(self._state[v] - (
                    self._nominal_mean[self.state_vars.index(v)]
                    if self._nominal_mean is not None else 0
                )),
            )
            self._anomaly_log.append(AnomalyEvent(
                timestamp=timestamp,
                channel=worst_var,
                value=self._state[worst_var],
                score=anomaly_score,
                threshold=self.anomaly_threshold,
            ))

        return entry

    def _compute_anomaly_score(self) -> float:
        """Mahalanobis distance from nominal operating point."""
        if self._nominal_mean is None:
            return 0.0
        x = np.array([self._state[v] for v in self.state_vars])
        diff = x - self._nominal_mean
        return float(np.sqrt(diff @ self._nominal_cov_inv @ diff))

    def generate_health_report(self) -> dict:
        """Return a summary of the current twin state and recent anomalies."""
        return {
            "current_state": dict(self._state),
            "anomaly_score": self._compute_anomaly_score(),
            "status": "ANOMALY" if self._compute_anomaly_score() > self.anomaly_threshold else "NOMINAL",
            "total_anomalies_logged": len(self._anomaly_log),
            "recent_anomalies": [
                {"timestamp": e.timestamp, "channel": e.channel,
                 "value": e.value, "score": round(e.score, 3)}
                for e in self._anomaly_log[-5:]
            ],
            "n_state_updates": len(self._history),
        }

    @property
    def history(self) -> list[dict]:
        return self._history
