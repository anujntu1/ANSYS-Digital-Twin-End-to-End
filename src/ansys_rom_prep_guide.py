"""
ANSYS ROM Prep workflow documentation and scripting utilities.

This module provides:
1. Documented workflow steps (comments as executable documentation)
2. PyAnsys scripting for automated ROM prep where possible
3. Result export helpers for post-processing

Workflow: ANSYS Workbench → Static ROM Prep → .rom → Twin Builder → .twin
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class WorkbenchSimulationConfig:
    """
    Configuration for an ANSYS Workbench simulation prepared for ROM export.

    Maps to the Parameter Set configuration in ANSYS Workbench that drives
    the design point study for ROM training data generation.
    """
    project_name: str
    simulation_type: str          # 'static_structural', 'fluent_cfd', 'ls_dyna'
    input_parameters: list[str]   # names must match Workbench parameter names
    output_parameters: list[str]  # names must match Workbench output parameters
    parameter_ranges: dict        # {param_name: [min, max]}
    n_design_points: int = 50     # number of design points for ROM training
    rom_prep_type: str = "static" # 'static' or 'dynamic'
    mesh_file: Optional[str] = None

    def to_json(self, path: str) -> str:
        """Export config to JSON for version control."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def from_json(cls, path: str) -> "WorkbenchSimulationConfig":
        with open(path) as f:
            return cls(**json.load(f))


class DesignPointManager:
    """
    Manage ANSYS Workbench design points for ROM training data generation.

    In ANSYS Workbench, ROM training requires running multiple "design points"
    (parameter combinations). This class helps plan and track those runs.
    """

    def __init__(self, config: WorkbenchSimulationConfig) -> None:
        self.config = config

    def generate_design_points(
        self, method: str = "lhs", seed: int = 42
    ) -> "list[dict]":
        """
        Generate the design point parameter table for ANSYS Workbench import.

        The output can be imported into Workbench's Parameter Set table
        via File → Import → CSV.

        Parameters
        ----------
        method : 'lhs' (Latin Hypercube), 'full_factorial', or 'random'

        Returns
        -------
        List of dicts — each dict is one design point (row in Workbench table)
        """
        n = self.config.n_design_points
        ranges = self.config.parameter_ranges
        names = list(ranges.keys())
        bounds = np.array(list(ranges.values()))

        if method == "lhs":
            from scipy.stats.qmc import LatinHypercube, scale
            s = LatinHypercube(d=len(names), seed=seed).random(n=n)
            samples = scale(s, bounds[:, 0], bounds[:, 1])
        elif method == "random":
            rng = np.random.default_rng(seed)
            samples = rng.uniform(bounds[:, 0], bounds[:, 1], (n, len(names)))
        else:
            raise ValueError(f"Unknown method: {method}")

        return [dict(zip(names, row)) for row in samples]

    def export_to_csv(self, design_points: list[dict], output_path: str) -> str:
        """
        Export design points to CSV for import into ANSYS Workbench Parameter Set.

        The CSV format matches the Workbench table: one row per design point,
        columns = parameter names.
        """
        import pandas as pd
        df = pd.DataFrame(design_points)
        df.to_csv(output_path, index=True, index_label="DesignPoint")
        return output_path


class TwinFileInspector:
    """
    Inspect a .twin file's metadata without evaluating it.
    Useful for version tracking and CI/CD pipeline integration.
    """

    @staticmethod
    def inspect(twin_file: str) -> dict:
        """
        Extract metadata from a .twin file (zip archive).

        .twin files are ZIP archives containing:
        - manifest.json  : model metadata
        - *.rom          : the ROM data
        - other assets
        """
        import zipfile
        path = Path(twin_file)
        if not path.exists():
            raise FileNotFoundError(f".twin file not found: {twin_file}")

        info = {
            "file": str(path),
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "contents": [],
        }

        with zipfile.ZipFile(twin_file, "r") as zf:
            info["contents"] = zf.namelist()
            # Try to read manifest if present
            for fname in zf.namelist():
                if fname.endswith("manifest.json") or fname == "manifest.json":
                    with zf.open(fname) as f:
                        try:
                            info["manifest"] = json.load(f)
                        except Exception:
                            pass

        return info

    @staticmethod
    def compare_twins(twin_a: str, twin_b: str) -> dict:
        """Compare two .twin files (e.g., after ROM update) to track changes."""
        a = TwinFileInspector.inspect(twin_a)
        b = TwinFileInspector.inspect(twin_b)
        return {
            "size_change_mb": round(b["size_mb"] - a["size_mb"], 3),
            "files_in_a_only": list(set(a["contents"]) - set(b["contents"])),
            "files_in_b_only": list(set(b["contents"]) - set(a["contents"])),
            "common_files": list(set(a["contents"]) & set(b["contents"])),
        }
