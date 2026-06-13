"""
PyTwin TwinModel wrapper for static ROM evaluation.

The ANSYS Twin Builder workflow exports a `.twin` file that encapsulates
a Reduced Order Model. PyTwin's TwinModel API evaluates this ROM at
any input parameter combination in milliseconds — replacing hours of
full FEA/CFD computation.

Workflow context
----------------
ANSYS Workbench (FEA/CFD)
    → Static ROM Prep (.rom export)
        → Twin Builder (compile to .twin)
            → PyTwin TwinModel (this module)
                → Python data pipeline / ML training

Supports: ANSYS 2024R2 / 2025R1 twin files
Requires: pytwin (pip install pytwin)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TwinModelEvaluator:
    """
    Wrapper around pytwin.TwinModel for convenient static ROM evaluation.

    A static ROM directly maps input parameters → output quantities
    (e.g. displacement → max stress, max deformation, reaction force).
    No time-stepping is needed.

    Parameters
    ----------
    twin_file : path to .twin file exported from ANSYS Twin Builder
    log_level : pytwin logging level (default 'WARNING')

    Example
    -------
    >>> evaluator = TwinModelEvaluator("rubber_press.twin")
    >>> evaluator.load()
    >>> result = evaluator.evaluate({"Displacement_mm": 5.0})
    >>> print(result)  # {"MaxStress_MPa": 12.4, "MaxDeformation_mm": 4.98}
    """

    def __init__(self, twin_file: str, log_level: str = "WARNING") -> None:
        self.twin_file = Path(twin_file)
        self.log_level = log_level
        self._model = None
        self._loaded = False

    def load(self) -> "TwinModelEvaluator":
        """Load the .twin file and initialise the PyTwin TwinModel."""
        try:
            from pytwin import TwinModel, modify_pytwin_logging, get_pytwin_log_file
            modify_pytwin_logging(new_level=self.log_level)
            self._model = TwinModel(model_filepath=str(self.twin_file))
            self._loaded = True
            logger.info(f"Loaded twin file: {self.twin_file.name}")
        except ImportError:
            raise ImportError(
                "pytwin is not installed. Run: pip install pytwin\n"
                "Or install ANSYS Python libraries: pip install ansys-pytwin"
            )
        return self

    def initialize(self, default_inputs: Optional[dict] = None) -> None:
        """
        Initialize twin model evaluation with default (or zero) inputs.
        Must be called once before evaluate().
        """
        self._check_loaded()
        inputs = default_inputs or {k: 0.0 for k in self.input_names}
        self._model.initialize_evaluation(inputs=inputs)

    def evaluate(self, inputs: dict[str, float]) -> dict[str, float]:
        """
        Evaluate the ROM at the given input parameter values.

        Parameters
        ----------
        inputs : dict of input_name → value

        Returns
        -------
        outputs : dict of output_name → predicted value
        """
        self._check_loaded()
        self._model.evaluate_twin_state(inputs=inputs)
        return dict(self._model.outputs)

    def parametric_sweep(
        self,
        param_name: str,
        values: np.ndarray,
        fixed_inputs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Sweep one input parameter over a range of values.

        Parameters
        ----------
        param_name   : name of the parameter to sweep
        values       : 1-D array of values to evaluate
        fixed_inputs : other inputs to hold constant

        Returns
        -------
        DataFrame with the swept parameter and all outputs
        """
        self._check_loaded()
        base = fixed_inputs.copy() if fixed_inputs else {k: 0.0 for k in self.input_names}
        self.initialize(base)

        records = []
        for v in values:
            inp = {**base, param_name: float(v)}
            out = self.evaluate(inp)
            records.append({param_name: v, **out})

        return pd.DataFrame(records)

    def grid_sweep(
        self,
        param_ranges: dict[str, np.ndarray],
        fixed_inputs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Full-factorial sweep over multiple parameters.

        Parameters
        ----------
        param_ranges : dict of param_name → array of values

        Returns
        -------
        DataFrame with all parameter combinations and outputs
        """
        self._check_loaded()
        import itertools
        base = fixed_inputs.copy() if fixed_inputs else {k: 0.0 for k in self.input_names}
        self.initialize(base)

        names = list(param_ranges.keys())
        grids = list(itertools.product(*[param_ranges[n] for n in names]))

        records = []
        for combo in grids:
            inp = {**base, **dict(zip(names, combo))}
            out = self.evaluate(inp)
            records.append({**dict(zip(names, combo)), **out})

        return pd.DataFrame(records)

    def generate_training_dataset(
        self,
        param_space: dict[str, tuple[float, float]],
        n_samples: int = 1000,
        method: str = "lhs",
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate a large dataset from the twin for ML training or DOE.

        Parameters
        ----------
        param_space : dict of param_name → (min, max)
        n_samples   : number of evaluation points
        method      : 'lhs' (Latin Hypercube) or 'random' or 'grid'

        Returns
        -------
        DataFrame with inputs + outputs — ready for ML training
        """
        self._check_loaded()
        rng = np.random.default_rng(seed)
        n_params = len(param_space)
        names = list(param_space.keys())
        bounds = np.array(list(param_space.values()))  # (n_params, 2)

        if method == "lhs":
            from scipy.stats.qmc import LatinHypercube, scale
            sampler = LatinHypercube(d=n_params, seed=seed)
            unit_samples = sampler.random(n=n_samples)
            samples = scale(unit_samples, bounds[:, 0], bounds[:, 1])
        elif method == "random":
            samples = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, n_params))
        else:
            raise ValueError(f"Unknown method: {method}")

        # Initialize with midpoint
        mid = {n: float((b[0] + b[1]) / 2) for n, b in zip(names, bounds)}
        self.initialize(mid)

        records = []
        for i, row in enumerate(samples):
            inp = dict(zip(names, row))
            out = self.evaluate(inp)
            records.append({**inp, **out})

            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i+1}/{n_samples} samples")

        df = pd.DataFrame(records)
        logger.info(f"Dataset complete: {len(df)} rows × {len(df.columns)} columns")
        return df

    @property
    def input_names(self) -> list[str]:
        self._check_loaded()
        return list(self._model.inputs.keys())

    @property
    def output_names(self) -> list[str]:
        self._check_loaded()
        return list(self._model.outputs.keys())

    @property
    def twin_info(self) -> dict:
        self._check_loaded()
        return {
            "twin_file": str(self.twin_file),
            "inputs": self.input_names,
            "outputs": self.output_names,
            "n_inputs": len(self.input_names),
            "n_outputs": len(self.output_names),
        }

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call load() before using the evaluator.")
