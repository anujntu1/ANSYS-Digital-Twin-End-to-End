"""
Demo: Data Center Thermal Digital Twin
========================================
Applies the ANSYS E2E workflow to data center cooling simulation:

  ANSYS Fluent (CFD) — server rack airflow + heat dissipation
      ↓ Static ROM Prep (inlet temp, power load → max server temp, PUE)
      ↓ Twin Builder → datacenter_thermal.twin
      ↓ PyTwin (TwinRuntime — dynamic response to varying workloads)
      ↓ Real-time cooling optimization

Demonstrates both static and dynamic twin evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

# ─── Synthetic data center thermal twin ─────────────────────────
# Replaces TwinModelEvaluator("datacenter_thermal.twin")
class DataCenterThermalTwin:
    """
    Surrogate for ANSYS Fluent CFD twin of server rack cooling.
    Inputs: IT power load (W), inlet air temp (°C), airflow rate (m³/s)
    Outputs: max server temp (°C), PUE, bypass fraction, COP
    """
    CP_AIR, RHO = 1006.0, 1.2

    def evaluate(self, power_w: float, t_inlet_c: float, flow_m3s: float) -> dict:
        mdot = flow_m3s * self.RHO
        delta_t = power_w / (mdot * self.CP_AIR)
        t_max = t_inlet_c + delta_t * 1.25  # 25% hotspot factor
        cop = power_w / (0.35 * power_w)  # COP ~ 2.86
        pue = 1.0 + (0.35 * power_w) / power_w
        bypass = max(0, 0.25 - 0.1 * flow_m3s)  # better flow = less bypass
        return {
            "MaxServerTemp_C": round(t_max, 2),
            "PUE": round(pue, 3),
            "CoolingCOP": round(cop, 3),
            "BypassFraction": round(bypass, 3),
        }

twin = DataCenterThermalTwin()

print("=" * 60)
print("DATA CENTER THERMAL TWIN — COOLING OPTIMIZATION")
print("=" * 60)

# ─── 1. Parametric sweep: power load vs airflow ─────────────────
print("\n1. DESIGN SPACE SWEEP")
powers = np.linspace(1000, 10000, 20)
flows  = np.linspace(0.2, 1.5, 15)
P, F = np.meshgrid(powers, flows)
T_max = np.array([[twin.evaluate(p, 22.0, f)["MaxServerTemp_C"]
                   for p in powers] for f in flows])

# ─── 2. Dynamic simulation: time-varying workload ────────────────
print("\n2. DYNAMIC RESPONSE — 8-HOUR WORKDAY SIMULATION")
hours = np.linspace(0, 8, 97)  # 5-min intervals
# Typical diurnal IT load profile
power_profile = (4000 + 3000 * np.sin(np.pi * hours / 8)
                 + 500 * np.sin(4 * np.pi * hours / 8)
                 + np.random.normal(0, 100, len(hours)))
inlet_profile  = 20 + 2 * np.sin(np.pi * hours / 8)   # ambient warming

results = []
for h, p, t_in in zip(hours, power_profile, inlet_profile):
    r = twin.evaluate(p, t_in, flow_m3s=0.8)
    results.append({"hour": h, "power_w": p, "t_inlet": t_in, **r})
df = pd.DataFrame(results)

print(f"  Max server temp peak: {df['MaxServerTemp_C'].max():.1f} °C")
print(f"  Average PUE: {df['PUE'].mean():.3f}")
print(f"  Hours above 40°C: {(df['MaxServerTemp_C'] > 40).sum() * 5 / 60:.1f} h")

# ─── 3. Cooling optimization ─────────────────────────────────────
print("\n3. AIRFLOW OPTIMIZATION — Minimize PUE subject to Tmax < 40°C")
from scipy.optimize import minimize_scalar

def pue_for_flow(flow, power, t_inlet):
    r = twin.evaluate(power, t_inlet, flow)
    if r["MaxServerTemp_C"] > 40.0:
        return 10.0  # penalty
    return r["PUE"]

peak_power = float(power_profile.max())
opt = minimize_scalar(lambda f: pue_for_flow(f, peak_power, 22.0),
                      bounds=(0.2, 2.0), method="bounded")
print(f"  Peak power: {peak_power:.0f} W")
print(f"  Optimal airflow: {opt.x:.3f} m³/s")
r_opt = twin.evaluate(peak_power, 22.0, opt.x)
print(f"  At optimal flow: {r_opt}")

# ─── 4. Training dataset for ML ──────────────────────────────────
print("\n4. GENERATING ML TRAINING DATASET")
from scipy.stats.qmc import LatinHypercube, scale as qmc_scale
sampler = LatinHypercube(d=3, seed=42)
pts = qmc_scale(sampler.random(2000), [500, 18, 0.2], [12000, 30, 2.0])
rows = [{"power_w": p, "t_inlet_c": t, "flow_m3s": f,
          **twin.evaluate(p, t, f)} for p, t, f in pts]
df_ml = pd.DataFrame(rows)
df_ml.to_csv("datacenter_thermal_training.csv", index=False)
print(f"  Saved: datacenter_thermal_training.csv ({len(df_ml)} rows)")

# ─── Plots ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Heatmap
ax = axes[0]
im = ax.pcolormesh(powers/1000, flows, T_max, cmap="hot_r", shading="auto")
plt.colorbar(im, ax=ax, label="Max Server Temp (°C)")
ax.contour(powers/1000, flows, T_max, levels=[40], colors=["white"],
           linewidths=2)
ax.set_xlabel("IT Power (kW)")
ax.set_ylabel("Airflow (m³/s)")
ax.set_title("Server Temp Map\n(white contour = 40°C limit)")

# Dynamic profile
ax = axes[1]
ax2 = ax.twinx()
ax.plot(df["hour"], df["MaxServerTemp_C"], color="tomato", lw=2, label="Max T (°C)")
ax.axhline(40, color="red", linestyle="--", alpha=0.6, label="40°C limit")
ax2.plot(df["hour"], df["power_w"]/1000, color="steelblue", lw=1.5,
         alpha=0.7, linestyle="--", label="IT Load (kW)")
ax.set_xlabel("Hour")
ax.set_ylabel("Temperature (°C)", color="tomato")
ax2.set_ylabel("IT Power (kW)", color="steelblue")
ax.set_title("8-Hour Workday Dynamic Response")
ax.legend(loc="upper left", fontsize=8)

# PUE scatter
ax = axes[2]
sc = ax.scatter(df_ml["power_w"]/1000, df_ml["PUE"],
                c=df_ml["MaxServerTemp_C"], cmap="RdYlGn_r",
                s=5, alpha=0.5)
plt.colorbar(sc, ax=ax, label="Max Server Temp (°C)")
ax.set_xlabel("IT Power (kW)")
ax.set_ylabel("PUE")
ax.set_title(f"PUE vs IT Load\n(N={len(df_ml)} ML training samples)")

fig.suptitle("Data Center Thermal Twin — ANSYS Fluent CFD → PyTwin",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
