"""
Demo: Rubber Press Machine Digital Twin (from Physics Simulation Document)
===========================================================================
Replicates the actual workflow documented at Pegatron Corporation:

  ANSYS Workbench (Static Structural)
      │  rubber DUT compressed by steel press head on hyperelastic pad
      ▼
  Static ROM Prep
      │  parameterized: Displacement → MaxStress, MaxDeformation, ReactionForce
      ▼
  Twin Builder → rubber_press.twin
      ▼
  PyTwin (this script)
      │  parametric sweep + dataset generation + validation
      ▼
  ML-ready training data (CSV)

NOTE: This demo uses synthetic data to simulate PyTwin outputs.
      Replace SyntheticTwinModel with TwinModelEvaluator when .twin file is available.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────────────────────────
# Synthetic twin model (replaces PyTwin when .twin file is present)
# Remove this and use TwinModelEvaluator for the real .twin file
# ─────────────────────────────────────────────────────────────────
class SyntheticTwinModel:
    """
    Simulates PyTwin TwinModel outputs for rubber press machine.
    Nonlinear rubber material (Mooney-Rivlin) response.
    Replace with: TwinModelEvaluator("rubber_press.twin")
    """
    # Mooney-Rivlin constants (typical silicone rubber)
    C1, C2 = 0.18, 0.03  # MPa

    def evaluate(self, displacement_mm: float) -> dict:
        d = displacement_mm / 100.0  # normalized stretch
        # Neo-Hookean stress ≈ 2*C1*(λ - 1/λ²) for uniaxial
        lam = 1 + d
        stress = 2 * (self.C1 + self.C2 / lam) * (lam - 1 / lam**2)
        deformation = displacement_mm * (1 - 0.05 * d)  # slight nonlinearity
        reaction_force = stress * 400  # N (400 mm² contact area)
        return {
            "MaxStress_MPa": round(stress, 4),
            "MaxDeformation_mm": round(deformation, 4),
            "ReactionForce_N": round(reaction_force, 2),
        }

twin = SyntheticTwinModel()

# ═══════════════════════════════════════════════════════════════
# SECTION 1: Parametric Sweep (replicates Twin Builder sweep)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("RUBBER PRESS MACHINE — DIGITAL TWIN EVALUATION")
print("=" * 60)
print("\n1. PARAMETRIC SWEEP: Displacement 0 → 20 mm")

displacements = np.linspace(0.1, 20.0, 40)
results = [{"Displacement_mm": d, **twin.evaluate(d)} for d in displacements]
df_sweep = pd.DataFrame(results)

print(df_sweep.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# SECTION 2: Large Dataset Generation (for ML training)
# ═══════════════════════════════════════════════════════════════
print("\n2. GENERATING ML TRAINING DATASET (500 samples, LHS)")

np.random.seed(42)
from scipy.stats.qmc import LatinHypercube, scale as qmc_scale
sampler = LatinHypercube(d=1, seed=42)
unit = sampler.random(n=500)
disps = qmc_scale(unit, 0.1, 20.0).flatten()

records = [{"Displacement_mm": d, **twin.evaluate(d)} for d in disps]
df_train = pd.DataFrame(records)
df_train.to_csv("rubber_press_training_data.csv", index=False)
print(f"  Saved: rubber_press_training_data.csv  ({len(df_train)} rows)")
print(f"\n  Output statistics:")
print(df_train[["MaxStress_MPa", "MaxDeformation_mm", "ReactionForce_N"]].describe().to_string())

# ═══════════════════════════════════════════════════════════════
# SECTION 3: Validation (compare twin vs reference FEA)
# ═══════════════════════════════════════════════════════════════
print("\n3. TWIN VALIDATION vs REFERENCE FEA")

# Simulate reference FEA data (with small random noise = measurement/mesh error)
ref_disps = np.linspace(1, 20, 20)
ref_data = []
for d in ref_disps:
    truth = twin.evaluate(d)
    # Add ±0.5% noise to simulate FEA reference
    noisy = {k: v * (1 + np.random.uniform(-0.005, 0.005)) for k, v in truth.items()}
    ref_data.append({"Displacement_mm": d, **noisy})
df_ref = pd.DataFrame(ref_data)

# Predict with twin
df_ref["Pred_MaxStress_MPa"] = df_ref["Displacement_mm"].apply(
    lambda d: twin.evaluate(d)["MaxStress_MPa"]
)

# Compute R²
from sklearn.metrics import r2_score
r2 = r2_score(df_ref["MaxStress_MPa"], df_ref["Pred_MaxStress_MPa"])
rmse = np.sqrt(((df_ref["MaxStress_MPa"] - df_ref["Pred_MaxStress_MPa"])**2).mean())
print(f"  MaxStress — R² = {r2:.5f},  RMSE = {rmse:.5f} MPa")
print(f"  {'PASS ✓' if r2 >= 0.999 else 'REVIEW'}: R² ≥ 0.999 required for deployment")

# ═══════════════════════════════════════════════════════════════
# SECTION 4: Design Optimization
# ═══════════════════════════════════════════════════════════════
print("\n4. DESIGN OPTIMIZATION: Find displacement for target stress")

TARGET_STRESS = 0.5  # MPa
from scipy.optimize import brentq

def stress_minus_target(d):
    return twin.evaluate(d)["MaxStress_MPa"] - TARGET_STRESS

d_optimal = brentq(stress_minus_target, 0.1, 20.0)
result_at_opt = twin.evaluate(d_optimal)
print(f"  Target stress: {TARGET_STRESS} MPa")
print(f"  Required displacement: {d_optimal:.3f} mm")
print(f"  Verification: {result_at_opt}")

# ═══════════════════════════════════════════════════════════════
# SECTION 5: Visualizations
# ═══════════════════════════════════════════════════════════════
print("\n5. Generating plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Stress-displacement
ax = axes[0]
ax.plot(df_sweep["Displacement_mm"], df_sweep["MaxStress_MPa"],
        "steelblue", lw=2, label="Twin prediction")
ax.scatter(df_ref["Displacement_mm"], df_ref["MaxStress_MPa"],
           color="tomato", s=40, zorder=5, label=f"FEA reference (R²={r2:.4f})")
ax.axvline(d_optimal, color="green", linestyle="--",
           label=f"Optimal: {d_optimal:.1f} mm")
ax.set_xlabel("Displacement (mm)")
ax.set_ylabel("Max Von Mises Stress (MPa)")
ax.set_title("Stress Response — Twin vs FEA")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Reaction force
ax = axes[1]
ax.plot(df_sweep["Displacement_mm"], df_sweep["ReactionForce_N"],
        "tomato", lw=2)
ax.fill_between(df_sweep["Displacement_mm"], df_sweep["ReactionForce_N"],
                alpha=0.15, color="tomato")
ax.set_xlabel("Displacement (mm)")
ax.set_ylabel("Reaction Force (N)")
ax.set_title("Reaction Force vs Displacement")
ax.grid(True, alpha=0.3)

# Training data distribution
ax = axes[2]
ax.hist(df_train["MaxStress_MPa"], bins=30, color="steelblue",
        edgecolor="white", alpha=0.8)
ax.set_xlabel("Max Stress (MPa)")
ax.set_ylabel("Count")
ax.set_title(f"ML Training Data Distribution\n(N={len(df_train)}, LHS sampling)")
ax.grid(True, alpha=0.3)

fig.suptitle("Rubber Press Machine — Digital Twin Evaluation\n"
             "ANSYS Workbench → Static ROM Prep → Twin Builder → PyTwin",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
print("\nDemo complete. Output: rubber_press_training_data.csv")
