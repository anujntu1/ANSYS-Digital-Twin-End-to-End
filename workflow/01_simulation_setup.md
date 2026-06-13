# Step 1: ANSYS Workbench Simulation Setup

## Overview
The first stage of the digital twin pipeline is a high-fidelity ANSYS simulation
configured for parametric ROM export.

## Tools
- **ANSYS 2025 R1 / 2024 R2** with Workbench, Static ROM Prep, Twin Builder

## Setup Steps

### 1.1 Define Input Parameters
In Workbench, promote simulation inputs to **Parameters**:
- Right-click a boundary condition value → "Make Input Parameter"
- Example: Press displacement, inlet flow rate, power load
- Each parameter appears in the **Parameter Set** table

### 1.2 Define Output Parameters  
Promote result quantities:
- Right-click a result probe → "Make Output Parameter"
- Examples: Max von Mises stress, max deformation, outlet temperature

### 1.3 Mesh for ROM
- Use a mesh that balances accuracy and computation time
- For ROM prep: ~50k–200k elements is typical for structural
- Finer mesh → more accurate ROM but longer design point runs

### 1.4 Run Baseline
Verify the simulation runs correctly at nominal parameters before
starting the design point study.

---
Next: [02_rom_prep.md](02_rom_prep.md)
