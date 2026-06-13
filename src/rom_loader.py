"""
ROM Loader — load and evaluate Reduced Order Models for digital twin applications.

Supports loading ROMs from:
  - JSON + numpy format (exported by rom-ansys-twin-builder)
  - Simple polynomial model definition
"""

from __future__ import annotations
import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import StandardScaler


@dataclass
class ROMMetadata:
    name: str
    rom_type: str
    n_dof: int
    n_modes: int
    energy_captured: float
    parameter_names: list[str]
    parameter_bounds: dict[str, dict]
    output_quantity: str


class ROMModel:
    """
    Load and evaluate a pre-built POD-RBF ROM.

    Parameters
    ----------
    rom_dir : path to ROM bundle directory (from twin_builder_export.py)
    """

    def __init__(self, rom_dir: str) -> None:
        self.rom_dir = rom_dir
        self._metadata: Optional[ROMMetadata] = None
        self._pod_modes: Optional[np.ndarray] = None
        self._pod_mean: Optional[np.ndarray] = None
        self._rbf: Optional[RBFInterpolator] = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_scale: Optional[np.ndarray] = None
        self._loaded = False

    def load(self) -> "ROMModel":
        """Load ROM bundle from disk."""
        meta_path = os.path.join(self.rom_dir, "rom_metadata.json")
        pod_path = os.path.join(self.rom_dir, "pod_basis.npz")
        surr_path = os.path.join(self.rom_dir, "surrogate_data.npz")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"ROM metadata not found: {meta_path}")

        with open(meta_path) as f:
            meta_raw = json.load(f)

        self._metadata = ROMMetadata(
            name=os.path.basename(self.rom_dir),
            rom_type=meta_raw["rom_type"],
            n_dof=meta_raw["n_dof"],
            n_modes=meta_raw["n_modes"],
            energy_captured=meta_raw["energy_captured"],
            parameter_names=list(meta_raw["parameter_space"].keys()),
            parameter_bounds=meta_raw["parameter_space"],
            output_quantity=meta_raw["output_quantity"],
        )

        pod_data = np.load(pod_path)
        self._pod_modes = pod_data["modes"]
        self._pod_mean = pod_data["mean"]

        surr_data = np.load(surr_path)
        self._rbf = RBFInterpolator(
            surr_data["param_train"],
            surr_data["alpha_train"],
            kernel=meta_raw.get("rbf_kernel", "thin_plate_spline"),
        )
        self._scaler_mean = np.array(meta_raw["scaler_mean"])
        self._scaler_scale = np.array(meta_raw["scaler_scale"])
        self._loaded = True
        return self

    def evaluate(self, params: dict[str, float]) -> np.ndarray:
        """
        Evaluate ROM at given parameter values.

        Parameters
        ----------
        params : dict mapping parameter name → value

        Returns
        -------
        full_field : np.ndarray, shape (n_dof,)
        """
        self._check_loaded()
        x = np.array([[params[k] for k in self._metadata.parameter_names]])
        x_scaled = (x - self._scaler_mean) / self._scaler_scale
        alpha = self._rbf(x_scaled)
        return (alpha @ self._pod_modes.T + self._pod_mean).squeeze()

    def scalar_outputs(self, params: dict[str, float]) -> dict[str, float]:
        """Return scalar summary statistics of the full-field prediction."""
        field = self.evaluate(params)
        return {
            "max": float(field.max()),
            "mean": float(field.mean()),
            "min": float(field.min()),
            "std": float(field.std()),
        }

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call load() first.")

    @property
    def metadata(self) -> ROMMetadata:
        self._check_loaded()
        return self._metadata


class ROMRegistry:
    """
    Manage multiple ROMs (e.g., thermal ROM, structural ROM) in a digital twin.

    Usage
    -----
    >>> registry = ROMRegistry()
    >>> registry.load_from_directory("./rom_bundles/")
    >>> thermal = registry.get("thermal_rom")
    >>> result = thermal.scalar_outputs({"power_w": 300, "mdot_kg_s": 0.5})
    """

    def __init__(self) -> None:
        self._roms: dict[str, ROMModel] = {}

    def register(self, name: str, rom: ROMModel) -> None:
        self._roms[name] = rom

    def load_from_directory(self, base_dir: str) -> None:
        """Load all ROM bundles found in subdirectories of base_dir."""
        if not os.path.isdir(base_dir):
            return
        for subdir in os.listdir(base_dir):
            full_path = os.path.join(base_dir, subdir)
            meta_file = os.path.join(full_path, "rom_metadata.json")
            if os.path.isdir(full_path) and os.path.exists(meta_file):
                try:
                    rom = ROMModel(full_path).load()
                    self._roms[subdir] = rom
                except Exception as e:
                    print(f"Warning: could not load ROM from {full_path}: {e}")

    def get(self, name: str) -> ROMModel:
        if name not in self._roms:
            raise KeyError(f"ROM '{name}' not found. Available: {list(self._roms.keys())}")
        return self._roms[name]

    def list_roms(self) -> list[str]:
        return list(self._roms.keys())
