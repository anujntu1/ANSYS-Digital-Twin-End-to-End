# ANSYS Digital Twin — End-to-End Pipeline

> Complete workflow from ANSYS FEA/CFD simulation to deployed digital twin,
> production-validated across multiple industrial domains at **Pegatron Corporation**.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python)](https://python.org)
[![ANSYS](https://img.shields.io/badge/ANSYS-2025R1%20%7C%202024R2-FFB500)](https://ansys.com)
[![PyTwin](https://img.shields.io/badge/PyTwin-TwinModel%20%7C%20TwinRuntime-00A3E0)](https://twin.docs.pyansys.com/)
[![Domains](https://img.shields.io/badge/Domains-Structural%20%7C%20CFD%20%7C%20LS--DYNA-green)]()

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANSYS WORKBENCH                               │
│  Static Structural │ ANSYS Fluent CFD │ LS-DYNA │ Thermal       │
│  (nonlinear FEA)   │ (multiphase VOF) │ (crash) │ (conjugate)   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Parameterized simulation
                           │ Design point study (LHS/DOE)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ANSYS STATIC ROM PREP                           │
│  50–200 design points → Reduced Order Model → .rom file         │
│  (seconds of FEA compressed to milliseconds of evaluation)      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ANSYS TWIN BUILDER                              │
│  .rom → compile → .twin (portable surrogate)                    │
│  Validation: R² > 0.999 vs reference FEA                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PYTWIN (this repository)                        │
│  TwinModel API   → static ROM: inputs → outputs instantly       │
│  TwinRuntime API → dynamic ROM: time-stepping for transients    │
│  • Parametric sweeps        • Validation vs reference FEA       │
│  • Large dataset generation • Sensitivity analysis (Sobol)      │
│  • Design optimization      • ML training data export           │
└─────────────────────────────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ML Models    REST APIs    Real-time
         (training)   (serving)   controllers
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              RUNTIME MONITORING (this repository)                │
│  DigitalTwinPipeline — sensor ingestion every time step         │
│  SensorDataProcessor — rolling stats, IQR outlier detection     │
│  DigitalTwinStateMonitor — sensor + ROM fusion                  │
│  Anomaly detection — Mahalanobis distance from nominal envelope  │
│  DataLogger — timestamped CSV output                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Domains Covered

| Domain | ANSYS Tool | ROM Inputs | Outputs |
|--------|-----------|------------|---------|
| **Rubber Press Machine** | ANSYS Mechanical (Static Structural) | Press displacement | Max stress, deformation, reaction force |
| **Data Center Cooling** | ANSYS Fluent | IT power, inlet temp, airflow | Max server temp, PUE, bypass fraction |
| **Glue Dispensing** | ANSYS Fluent (VOF multiphase) | Flow rate, viscosity, nozzle position | Bead shape, fill uniformity, pressure |
| **Aerodynamics** | ANSYS Fluent | Velocity, angle of attack | Drag (Cd), lift (Cl), pressure |
| **Thermal-Structural** | ANSYS Mechanical (Coupled) | Power, ambient temp | Thermal stress, junction temp |
| **Crash Analysis** | LS-DYNA | Impact velocity, material props | Peak stress, energy absorption, HIC |

---

## Features

### Build Pipeline (ANSYS → PyTwin)
- **`TwinModelEvaluator`** — PyTwin TwinModel wrapper with parametric sweep, LHS dataset generation, Sobol sensitivity
- **`TwinRuntimeEvaluator`** — PyTwin TwinRuntime wrapper for dynamic (time-stepping) ROMs
- **`ROMDataGenerator`** — Large-scale dataset factory from .twin files for ML training
- **`TwinValidator`** — R², RMSE, MAPE validation vs reference FEA/CFD with parity plots
- **`ROMModel` / `ROMRegistry`** — Load and evaluate POD-RBF ROMs exported from Twin Builder
- **Workflow docs** (`workflow/`) — step-by-step ANSYS → PyTwin pipeline guide

### Runtime Monitoring (Deploy → Monitor)
- **`DigitalTwinPipeline`** — Full runtime orchestrator: sensor ingestion → ROM evaluation → state fusion → anomaly detection → CSV logging
- **`DigitalTwinStateMonitor`** — Mahalanobis-distance anomaly detection, sensor-ROM state fusion, health reporting
- **`SensorDataProcessor`** — Rolling-window statistics, IQR outlier detection per sensor channel
- **`SensorDataNormalizer`** — z-score / min-max normalisation for sensor streams
- **`FeatureExtractor`** — Time-domain statistical features (RMS, kurtosis, skewness, crest factor) for ML
- **`configs/pipeline_config.yaml`** — YAML configuration for all pipeline parameters

---

## Installation

```bash
git clone https://github.com/anujntu1/ansys-digital-twin-e2e.git
cd ansys-digital-twin-e2e
pip install -r requirements.txt
```

ANSYS PyTwin:
```bash
pip install pytwin
```

---

## Quick Start (with a real .twin file)

```python
from src.twin_model_evaluator import TwinModelEvaluator

# Load your .twin file
evaluator = TwinModelEvaluator("rubber_press.twin")
evaluator.load()
evaluator.initialize()

print(evaluator.twin_info)
# → {'inputs': ['Displacement_mm'], 'outputs': ['MaxStress_MPa', ...]}

# Single evaluation
result = evaluator.evaluate({"Displacement_mm": 10.0})
print(result)
# → {"MaxStress_MPa": 8.23, "MaxDeformation_mm": 9.97}

# Generate 5000-sample ML training dataset
df = evaluator.generate_training_dataset(
    param_space={"Displacement_mm": (0.0, 20.0)},
    n_samples=5000,
    method="lhs",
)
df.to_csv("training_data.csv", index=False)
```

---

## Run the Demos

```bash
# Rubber press machine (mechanical structural domain)
python examples/rubber_press_twin_demo.py

# Data center thermal (CFD domain)
python examples/data_center_twin_demo.py

# Runtime monitoring: sensor fusion + anomaly detection (server rack twin)
python examples/twin_pipeline_demo.py
```

---

## Workflow Documentation

| Step | File | Description |
|------|------|-------------|
| 1 | [workflow/01_simulation_setup.md](workflow/01_simulation_setup.md) | ANSYS Workbench parametric setup |
| 2 | [workflow/02_rom_prep.md](workflow/02_rom_prep.md) | Static ROM Prep — design points → .rom |
| 3 | [workflow/03_twin_builder.md](workflow/03_twin_builder.md) | Twin Builder — .rom → .twin |
| 4 | [workflow/04_pytwin_deployment.md](workflow/04_pytwin_deployment.md) | PyTwin API — Python integration |

---

## Author

**Anuj Chauhan, Ph.D.**  
Section Manager, Software R&D — Pegatron Corporation  
[LinkedIn](https://www.linkedin.com/in/anuj-chauhan-phd-5a8ba411a/) | anujntuchem@gmail.com
