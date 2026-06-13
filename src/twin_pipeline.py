"""
Digital Twin Pipeline — orchestrates sensor ingestion, ROM evaluation,
state updates, anomaly detection, and logging.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .state_monitor import DigitalTwinStateMonitor, SensorDataProcessor
from .data_preprocessor import SensorDataNormalizer, DataLogger


@dataclass
class PipelineConfig:
    state_variables: list[str]
    sensor_channels: list[str]
    anomaly_threshold: float = 3.0
    sensor_weight: float = 0.7
    rolling_window: int = 50
    log_output_path: str = "twin_log.csv"
    sampling_rate_hz: float = 1.0


class DigitalTwinPipeline:
    """
    End-to-end physics-based digital twin pipeline.

    Architecture
    ------------
    Raw sensor data
         │
         ▼ SensorDataProcessor (rolling stats, outlier detection)
         │
         ▼ SensorDataNormalizer (z-score normalisation)
         │
         ├──────────────────────────────────┐
         │                                  ▼
         │                        ROM evaluation
         │                        (physics prediction)
         │                                  │
         ▼                                  ▼
    DigitalTwinStateMonitor ←── fuse sensor + ROM ───┘
         │
         ▼ anomaly detection (Mahalanobis distance)
         │
         ▼ DataLogger (timestamped CSV)

    Parameters
    ----------
    config : PipelineConfig
    rom_evaluator : callable (params_dict) → predicted_state_dict, or None
    """

    def __init__(
        self,
        config: PipelineConfig,
        rom_evaluator: Optional[Callable[[dict], dict]] = None,
    ) -> None:
        self.config = config
        self.rom_evaluator = rom_evaluator

        self._monitor = DigitalTwinStateMonitor(
            state_variables=config.state_variables,
            anomaly_threshold=config.anomaly_threshold,
        )
        self._sensor_proc = SensorDataProcessor(window=config.rolling_window)
        self._normalizer = SensorDataNormalizer()
        self._logger = DataLogger()
        self._step_count = 0
        self._initialized = False

    def initialize(self, warmup_data: np.ndarray, sensor_channel_order: list[str]) -> None:
        """
        Calibrate normaliser and anomaly detector using warmup data.

        Parameters
        ----------
        warmup_data          : (n_samples, n_channels) — nominal operating data
        sensor_channel_order : ordered list of channel names matching columns
        """
        self._normalizer.fit(warmup_data)

        # Build nominal state from warmup (assume sensor channels = state variables)
        n = len(self.config.state_variables)
        n_cols = min(n, warmup_data.shape[1])
        nominal_state = warmup_data[:, :n_cols]
        self._monitor.calibrate_nominal(nominal_state)
        self._initialized = True

    def run_step(
        self,
        timestamp: float,
        sensor_data: dict[str, float],
    ) -> dict:
        """
        Process one time step of the digital twin.

        Parameters
        ----------
        timestamp   : current simulation time (seconds)
        sensor_data : dict of channel → measured value

        Returns
        -------
        result dict with state, anomaly score, processed sensor stats
        """
        # 1. Pre-process sensor readings
        processed = {
            ch: self._sensor_proc.update(ch, v)
            for ch, v in sensor_data.items()
        }

        # 2. Evaluate ROM (if available)
        rom_predictions = None
        if self.rom_evaluator is not None:
            try:
                rom_predictions = self.rom_evaluator(sensor_data)
            except Exception as e:
                print(f"[Twin Pipeline] ROM evaluation failed at t={timestamp}: {e}")

        # 3. Update state monitor
        state_entry = self._monitor.update_state(
            timestamp=timestamp,
            sensor_readings=sensor_data,
            rom_predictions=rom_predictions,
            sensor_weight=self.config.sensor_weight,
        )

        # 4. Log
        self._logger.append(timestamp, state_entry)
        self._step_count += 1

        return {
            "timestamp": timestamp,
            "step": self._step_count,
            "state": dict(self._monitor._state),
            "anomaly_score": state_entry["anomaly_score"],
            "is_anomaly": state_entry["is_anomaly"],
            "outlier_channels": [
                ch for ch, p in processed.items() if p.get("is_outlier", False)
            ],
        }

    def generate_health_report(self) -> dict:
        report = self._monitor.generate_health_report()
        report["steps_processed"] = self._step_count
        return report

    def save_log(self, path: Optional[str] = None) -> str:
        out = path or self.config.log_output_path
        self._logger.save_csv(out)
        return out
