# Step 2: Static ROM Prep — Generating the .rom File

## Overview
ANSYS Static ROM Prep runs multiple design points (parameter combinations)
to build the training dataset for the Reduced Order Model.

## Workflow

### 2.1 Configure the Parameter Set
In Workbench Parameter Set:
```
Input Parameters:    Displacement [0–20 mm]
Output Parameters:   MaxStress, MaxDeformation, ReactionForce
```

### 2.2 Set Up Design Points
- Use **Design of Experiments (DOE)** to generate sampling points
- Recommended: Latin Hypercube Sampling (LHS), 30–100 design points
- See `src/ansys_rom_prep_guide.py` for Python-assisted DOE generation

### 2.3 Run All Design Points
- Workbench runs each design point sequentially or in parallel (HPC)
- Results are stored in the Parameter Set table
- Typical runtime: 1–10 min per point depending on mesh size

### 2.4 Launch Static ROM Prep
- In Workbench toolbox, drag **Static ROM Prep** into the project
- Connect it to the **Parameter Set** outputs
- Configure: select input/output parameters, set polynomial order
- Click **Update** → generates `.rom` file

### 2.5 Export .rom File
- Right-click ROM Prep → **Export ROM**
- Save as `project_name.rom`

---
Next: [03_twin_builder.md](03_twin_builder.md)
