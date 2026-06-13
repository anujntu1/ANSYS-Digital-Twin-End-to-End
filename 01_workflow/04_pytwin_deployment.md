# Step 4: PyTwin Deployment — Python Integration

## Overview
The `.twin` file is loaded and evaluated using the **PyTwin** Python package.
This is the final deployment stage — enabling real-time ROM evaluation from
any Python environment (scripts, REST APIs, ML pipelines, dashboards).

## Installation
```bash
pip install pytwin
```

## Static ROM (TwinModel API)

```python
from pytwin import TwinModel

# Load the .twin file
tm = TwinModel(model_filepath="rubber_press.twin")

# Initialize (set starting input values)
tm.initialize_evaluation(inputs={"Displacement_mm": 0.0})

# Evaluate at a specific operating point
tm.evaluate_twin_state(inputs={"Displacement_mm": 10.0})

# Read outputs
print(tm.outputs)
# → {"MaxStress_MPa": 8.23, "MaxDeformation_mm": 9.97, "ReactionForce_N": 3291.0}
```

## Dynamic ROM (TwinRuntime API)

```python
from pytwin import TwinRuntime

rt = TwinRuntime(model_filepath="thermal_transient.twin")
rt.twin_initialize()

# Step through time
for t, power in zip(time_steps, power_profile):
    rt.twin_simulate(time_step_size=t, inputs={"PowerLoad_W": power})
    outputs = rt.twin_get_outputs()
    print(f"t={t:.2f}s: T_max={outputs['MaxTemp_C']:.2f}°C")
```

## Introspection

```python
# List available inputs/outputs
print(tm.inputs)    # dict of input_name → current value
print(tm.outputs)   # dict of output_name → current value

# For TwinRuntime
print(rt.twin_get_input_names())
print(rt.twin_get_output_names())
print(rt.twin_get_sdk_version())
```

## Integration with this Toolkit

See `src/twin_model_evaluator.py` — a production-ready wrapper that adds:
- Parametric sweeps
- LHS-sampled dataset generation
- Validation against reference FEA
- Sobol sensitivity analysis
- CSV/Parquet export

---
See also: [PyTwin documentation](https://twin.docs.pyansys.com/)
