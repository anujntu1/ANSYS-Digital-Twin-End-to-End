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
| **Static Structural** | ANSYS Mechanical | Displacement, load, material props | Max stress, deformation, reaction force |
| **Thermal Management** | ANSYS Fluent / Icepak | Power, inlet temp, airflow | Max temperature, PUE, bypass fraction |
| **Multiphase CFD** | ANSYS Fluent (VOF) | Flow rate, viscosity, geometry params | Phase interface, pressure, fill quality |
| **External Aerodynamics** | ANSYS Fluent | Velocity, angle of attack, ride height | Drag (Cd), lift (Cl), pressure distribution |
| **Thermal-Structural** | ANSYS Mechanical (Coupled) | Power, ambient temp | Thermal stress, junction temperature |
| **High-Impact Dynamics** | LS-DYNA | Impact velocity, material props | Peak stress, energy absorption, HIC |

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

# Load your .twin file exported from ANSYS Twin Builder
evaluator = TwinModelEvaluator("your_model.twin")
evaluator.load()
evaluator.initialize()

print(evaluator.twin_info)
# → {'inputs': ['input_param_1', 'input_param_2'], 'outputs': ['output_1', 'output_2']}

# Single evaluation — millisecond speed
result = evaluator.evaluate({"input_param_1": 10.0, "input_param_2": 25.0})
print(result)
# → {"output_1": 8.23, "output_2": 9.97}

# Generate large ML training dataset via LHS sampling
df = evaluator.generate_training_dataset(
    param_space={"input_param_1": (0.0, 20.0), "input_param_2": (10.0, 50.0)},
    n_samples=5000,
    method="lhs",
)
df.to_csv("training_data.csv", index=False)
```

---

## Run the Demos

```bash
# Static structural domain — parametric twin evaluation
python examples/rubber_press_twin_demo.py

# CFD thermal domain — data center cooling twin
python examples/data_center_twin_demo.py

# Runtime monitoring — sensor fusion + anomaly detection
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
Software R&D — Pegatron Corporation  
[LinkedIn](https://www.linkedin.com/in/anuj-chauhan-phd-5a8ba411a/) | anujntuchem@gmail.com
