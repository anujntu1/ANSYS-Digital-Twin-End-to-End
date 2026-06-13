"""
PyTwin TwinRuntime wrapper for dynamic ROM time-stepping.

TwinRuntime handles dynamic ROMs where outputs evolve over time —
used for transient thermal, vibration, or fluid simulations where
the twin must step forward in time.

Static vs Dynamic ROM:
    TwinModel   → static ROM: inputs → outputs (no time)
    TwinRuntime → dynamic ROM: inputs evolve over time steps

Use cases at Pegatron:
    - Transient thermal response of server hardware under varying workloads
    - Time-varying structural loading (fatigue cycling, thermal cycling)
    - Dynamic data center cooling response to load changes
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TwinRuntimeEvaluator:
    """
    Wrapper around pytwin.TwinRuntime for dynamic (time-stepping) ROM evaluation.

    Parameters
    ----------
    twin_file : path to .twin file (must contain a dynamic ROM)
    """

    def __init__(self, twin_file: str) -> None:
        self.twin_file = Path(twin_file)
        self._runtime = None
        self._loaded = False
        self._time = 0.0
        self._history: list[dict] = []

    def load(self, log_level: str = "WARNING") -> "TwinRuntimeEvaluator":
        """Load and initialize the TwinRuntime."""
        try:
            from pytwin import TwinRuntime, modify_pytwin_logging
            modify_pytwin_logging(new_level=log_level)
            self._runtime = TwinRuntime(model_filepath=str(self.twin_file))
            self._runtime.twin_initialize()
            self._loaded = True
        except ImportError:
            raise ImportError("Run: pip install pytwin")
        return self

    def step(
        self,
        time_step: float,
        inputs: dict[str, float],
    ) -> dict[str, float]:
        """
        Advance the twin by one time step.

        Parameters
        ----------
        time_step : duration of this step (seconds)
        inputs    : dict of input_name → value for this step

        Returns
        -------
        outputs   : dict of output_name → current value
        """
        self._check_loaded()
        self._runtime.twin_simulate(time_step_size=time_step, inputs=inputs)
        self._time += time_step
        outputs = dict(self._runtime.twin_get_outputs())
        self._history.append({"time": self._time, **inputs, **outputs})
        return outputs

    def run_time_series(
        self,
        time_steps: np.ndarray,
        input_series: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Run the twin over a full time series.

        Parameters
        ----------
        time_steps    : (N,) array of time step durations (s)
        input_series  : dict of input_name → (N,) array of values

        Returns
        -------
        DataFrame with time + inputs + outputs at each step
        """
        self._check_loaded()
        n = len(time_steps)
        for i in range(n):
            inputs = {name: float(arr[i]) for name, arr in input_series.items()}
            self.step(time_steps[i], inputs)
            if (i + 1) % 50 == 0:
                logger.info(f"Step {i+1}/{n} — t={self._time:.3f}s")

        return pd.DataFrame(self._history)

    def reset(self) -> None:
        """Reset the twin to its initial state (t=0)."""
        self._check_loaded()
        self._runtime.twin_initialize()
        self._time = 0.0
        self._history.clear()

    @property
    def current_time(self) -> float:
        return self._time

    @property
    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    @property
    def input_names(self) -> list[str]:
        self._check_loaded()
        return list(self._runtime.twin_get_input_names())

    @property
    def output_names(self) -> list[str]:
        self._check_loaded()
        return list(self._runtime.twin_get_output_names())

    def inspect(self) -> dict:
        """Print twin model metadata."""
        self._check_loaded()
        return {
            "twin_file": str(self.twin_file),
            "inputs": self.input_names,
            "outputs": self.output_names,
            "sdk_version": self._runtime.twin_get_sdk_version(),
        }

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call load() first.")
