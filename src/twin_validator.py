"""
Twin model validation: compare .twin ROM outputs to reference FEA/CFD data.

Validation is a critical step before deploying any ROM — it confirms
that the twin accurately represents the original simulation across
the full design space, not just at training points.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


class TwinValidator:
    """
    Validate a .twin ROM against reference simulation data.

    Computes standard ROM validation metrics:
    - R² (coefficient of determination)
    - RMSE (root mean square error)
    - Max absolute error
    - Mean absolute percentage error (MAPE)
    - Parity plots for each output

    Parameters
    ----------
    twin_evaluator : loaded TwinModelEvaluator instance
    """

    def __init__(self, twin_evaluator) -> None:
        self.evaluator = twin_evaluator

    def validate_from_csv(self, reference_csv: str, input_cols: list[str],
                           output_cols: list[str]) -> dict:
        """
        Validate twin against reference data from a CSV file.

        The CSV should contain both input parameters and reference output values
        (typically exported from a full ANSYS simulation run).
        """
        ref = pd.read_csv(reference_csv)
        return self.validate_from_dataframe(ref, input_cols, output_cols)

    def validate_from_dataframe(
        self,
        reference_df: pd.DataFrame,
        input_cols: list[str],
        output_cols: list[str],
    ) -> dict:
        """
        Validate twin against reference DataFrame.

        Returns
        -------
        dict with metrics per output column
        """
        self.evaluator.initialize({c: 0.0 for c in input_cols})

        predictions = []
        for _, row in reference_df.iterrows():
            inp = {c: float(row[c]) for c in input_cols}
            pred = self.evaluator.evaluate(inp)
            predictions.append(pred)

        pred_df = pd.DataFrame(predictions)
        metrics = {}

        for col in output_cols:
            if col not in pred_df.columns:
                continue
            ref_vals = reference_df[col].values.astype(float)
            pred_vals = pred_df[col].values.astype(float)
            metrics[col] = self._compute_metrics(ref_vals, pred_vals)

        return metrics

    def _compute_metrics(self, reference: np.ndarray, predicted: np.ndarray) -> dict:
        residuals = predicted - reference
        ss_res = (residuals**2).sum()
        ss_tot = ((reference - reference.mean())**2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        rmse = float(np.sqrt((residuals**2).mean()))
        mae = float(np.abs(residuals).mean())
        max_err = float(np.abs(residuals).max())
        mape = float((np.abs(residuals) / (np.abs(reference) + 1e-12)).mean() * 100)

        return {
            "R2": round(float(r2), 5),
            "RMSE": round(rmse, 5),
            "MAE": round(mae, 5),
            "MaxError": round(max_err, 5),
            "MAPE_%": round(mape, 3),
            "n_points": len(reference),
        }

    def parity_plot(
        self,
        reference_df: pd.DataFrame,
        input_cols: list[str],
        output_cols: list[str],
        figsize: tuple = (6 * 2, 5),
    ) -> plt.Figure:
        """Generate parity plots (predicted vs reference) for all outputs."""
        self.evaluator.initialize({c: 0.0 for c in input_cols})

        predictions = []
        for _, row in reference_df.iterrows():
            inp = {c: float(row[c]) for c in input_cols}
            pred = self.evaluator.evaluate(inp)
            predictions.append(pred)
        pred_df = pd.DataFrame(predictions)

        n = len(output_cols)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, col in zip(axes, output_cols):
            if col not in pred_df.columns:
                continue
            ref = reference_df[col].values
            pred = pred_df[col].values
            lims = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
            ax.scatter(ref, pred, alpha=0.6, s=20, edgecolors="k", linewidth=0.3)
            ax.plot(lims, lims, "r--", lw=1.5, label="Perfect")
            r2 = 1 - ((pred - ref)**2).sum() / (((ref - ref.mean())**2).sum() + 1e-12)
            ax.set_title(f"{col}\nR² = {r2:.4f}")
            ax.set_xlabel("Reference (FEA/CFD)")
            ax.set_ylabel("Twin Prediction")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Twin Model Validation — Parity Plots", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    def print_validation_report(self, metrics: dict) -> None:
        """Print formatted validation report to console."""
        print(f"\n{'═'*55}")
        print(f"  TWIN MODEL VALIDATION REPORT")
        print(f"{'═'*55}")
        for output, m in metrics.items():
            print(f"\n  Output: {output}")
            print(f"  {'─'*45}")
            for k, v in m.items():
                status = ""
                if k == "R2":
                    status = "✓ PASS" if v >= 0.99 else ("⚠ CHECK" if v >= 0.95 else "✗ FAIL")
                elif k == "MAPE_%":
                    status = "✓ PASS" if v <= 1.0 else ("⚠ CHECK" if v <= 5.0 else "✗ FAIL")
                print(f"    {k:<15}: {v:>12}  {status}")
        print(f"\n{'═'*55}")
