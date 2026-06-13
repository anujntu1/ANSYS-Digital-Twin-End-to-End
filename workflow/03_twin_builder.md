# Step 3: Twin Builder — Compiling the .twin File

## Overview
ANSYS Twin Builder takes the `.rom` file and compiles it into a deployable
`.twin` file — a self-contained portable surrogate model.

## Workflow

### 3.1 Open Twin Builder
- From Workbench: double-click **Twin Builder** block
- Or: open standalone from ANSYS 2025 R1

### 3.2 Import the .rom File
- In Twin Builder schematic: **Tools → Static ROM**
- Browse and select the `.rom` file from Step 2
- A ROM block appears with input/output ports

### 3.3 Configure ROM Block
The block exposes:
- **Input ports**: simulation input parameters (e.g. `Displacement_mm`)
- **Output ports**: simulation output parameters (e.g. `MaxStress_MPa`)

### 3.4 Validate ROM in Twin Builder
- Select **ROM Validation** view
- Load the original design point data as reference
- Twin Builder plots predicted vs reference — check R² > 0.99

### 3.5 Compile as Twin Model
- Right-click ROM block → **Compile Twin Model**
- Set solver type and step parameters if needed
- Click **OK**

### 3.6 Export .twin File
- **File → Export Twin** → save as `project_name.twin`
- The `.twin` file is a portable ZIP archive containing:
  - `manifest.json` — metadata, input/output names
  - ROM data files
  - Model configuration

---
Next: [04_pytwin_deployment.md](04_pytwin_deployment.md)
