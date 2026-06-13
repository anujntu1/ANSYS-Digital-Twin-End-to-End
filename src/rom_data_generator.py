"""
Large-scale data generation from ANSYS Twin files.

Generates training datasets, design space exploration data, and
validation datasets by evaluating the deployed .twin ROM over
thousands of input combinations — replacing equivalent full FEA/CFD runs.

This is the "data factory" component of the digital twin workflow:
the .twin file acts as a fast surrogate, enabling ML model training,
sensitivity analysis, and design optimization at scale.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataGenerationConfig:
    """Configuration for ROM-based dataset generation."""
    twin_file: str
    output_dir: str = "./generated_data"
    n_samples: int = 5000
    sampling_method: str = "lhs"          # 'lhs', 'random', 'sobol'
    seed: int = 42
    save_format: str = "csv"              # 'csv', 'parquet', 'json'
    batch_size: int = 500                  # evaluate in batches (memory efficiency)
    validation_split: float = 0.15
    param_space: dict = field(default_factory=dict)


class ROMDataGenerator:
    """
    Generate large datasets from a .twin file for ML/AI training,
    sensitivity analysis, and optimization.

    This class is the bridge between ANSYS simulation and Python data
    science workflows — enabling training of surrogate ML models that
    can replace even the ROM for edge deployment.

    Example workflow
    ----------------
    Twin file (ms/eval) → ROMDataGenerator (1000s of evals/sec)
        → training_data.csv
            → scikit-learn / PyTorch model
                → edge device / real-time controller
    """

    def __init__(self, config: DataGenerationConfig) -> None:
        self.config = config
        self._evaluator = None
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_evaluator(self):
        if self._evaluator is None:
            from .twin_model_evaluator import TwinModelEvaluator
            self._evaluator = TwinModelEvaluator(self.config.twin_file)
            self._evaluator.load()
        return self._evaluator

    def generate(self, verbose: bool = True) -> pd.DataFrame:
        """
        Generate the full dataset and save to disk.

        Returns
        -------
        DataFrame with all input + output columns
        """
        ev = self._get_evaluator()
        if verbose:
            print(f"Twin model: {Path(self.config.twin_file).name}")
            print(f"Inputs: {ev.input_names}")
            print(f"Outputs: {ev.output_names}")
            print(f"Generating {self.config.n_samples} samples ({self.config.sampling_method} sampling)...")

        df = ev.generate_training_dataset(
            param_space=self.config.param_space,
            n_samples=self.config.n_samples,
            method=self.config.sampling_method,
            seed=self.config.seed,
        )

        # Train/validation split
        n_val = int(len(df) * self.config.validation_split)
        df["split"] = "train"
        df.iloc[-n_val:, df.columns.get_loc("split")] = "val"

        # Save
        self._save(df, "full_dataset")
        self._save(df[df["split"] == "train"].drop(columns="split"), "train")
        self._save(df[df["split"] == "val"].drop(columns="split"), "val")

        if verbose:
            self._print_summary(df)

        return df

    def sensitivity_analysis(
        self,
        param_ranges: dict[str, tuple[float, float]],
        n_levels: int = 20,
        baseline: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        One-at-a-time (OAT) sensitivity analysis: vary each parameter
        while holding others at baseline, observe output change.

        Returns
        -------
        DataFrame with sensitivity index per (input, output) pair
        """
        ev = self._get_evaluator()
        baseline = baseline or {k: (v[0] + v[1]) / 2 for k, v in param_ranges.items()}
        ev.initialize(baseline)

        rows = []
        for param, (lo, hi) in param_ranges.items():
            values = np.linspace(lo, hi, n_levels)
            sweep = ev.parametric_sweep(param, values, fixed_inputs=baseline)
            for _, row in sweep.iterrows():
                row_dict = row.to_dict()
                row_dict["varied_param"] = param
                rows.append(row_dict)

        df = pd.DataFrame(rows)
        return df

    def sobol_indices(
        self,
        param_space: dict[str, tuple[float, float]],
        n_base: int = 256,
        seed: int = 42,
    ) -> dict[str, dict]:
        """
        Estimate first-order Sobol sensitivity indices via the Saltelli method.

        Indicates which inputs have the most influence on each output.
        Useful for ROM dimensionality reduction.

        Returns
        -------
        dict of output_name → {input_name: S1 index}
        """
        from scipy.stats.qmc import Sobol, scale as qmc_scale

        ev = self._get_evaluator()
        names = list(param_space.keys())
        bounds = np.array(list(param_space.values()))
        n_params = len(names)

        sampler = Sobol(d=2 * n_params, scramble=True, seed=seed)
        raw = sampler.random_base2(m=int(np.log2(n_base)))  # (n_base, 2*n_params)

        A_unit = raw[:, :n_params]
        B_unit = raw[:, n_params:]
        A = qmc_scale(A_unit, bounds[:, 0], bounds[:, 1])
        B = qmc_scale(B_unit, bounds[:, 0], bounds[:, 1])

        mid = {n: float((b[0] + b[1]) / 2) for n, b in zip(names, bounds)}
        ev.initialize(mid)

        def eval_matrix(mat):
            return pd.DataFrame([ev.evaluate(dict(zip(names, row))) for row in mat])

        fA = eval_matrix(A)
        fB = eval_matrix(B)

        sobol = {}
        for out in fA.columns:
            yA = fA[out].values
            yB = fB[out].values
            var_total = np.var(np.concatenate([yA, yB]))
            s1 = {}
            for i, name in enumerate(names):
                AB_i = A.copy()
                AB_i[:, i] = B[:, i]
                fABi = eval_matrix(AB_i)[out].values
                s1[name] = float(np.mean(yB * (fABi - yA)) / (var_total + 1e-12))
            sobol[out] = s1

        return sobol

    def _save(self, df: pd.DataFrame, name: str) -> str:
        out = Path(self.config.output_dir)
        if self.config.save_format == "csv":
            path = out / f"{name}.csv"
            df.to_csv(path, index=False)
        elif self.config.save_format == "parquet":
            path = out / f"{name}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = out / f"{name}.json"
            df.to_json(path, orient="records", indent=2)
        return str(path)

    def _print_summary(self, df: pd.DataFrame) -> None:
        print(f"\n{'─'*50}")
        print(f"  Dataset summary")
        print(f"{'─'*50}")
        print(f"  Total samples : {len(df)}")
        print(f"  Train / Val   : {(df['split']=='train').sum()} / {(df['split']=='val').sum()}")
        print(f"  Columns       : {list(df.columns)}")
        print(f"  Saved to      : {self.config.output_dir}/")
        print(f"{'─'*50}")
        print(df.describe().to_string())
