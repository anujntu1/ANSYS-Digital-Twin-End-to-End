"""
Demo: Digital Twin Pipeline
============================
Simulates 200 time steps of a server rack digital twin:
  - Synthetic sensor data (temperature, power, airflow)
  - Simple polynomial ROM for temperature prediction
  - Anomaly injection at step 150
  - Full state monitoring, anomaly detection, and logging
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.twin_pipeline import DigitalTwinPipeline, PipelineConfig
from src.data_preprocessor import FeatureExtractor

import matplotlib.pyplot as plt

np.random.seed(7)

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
config = PipelineConfig(
    state_variables=["temperature_c", "power_w", "airflow_m3s"],
    sensor_channels=["temperature_c", "power_w", "airflow_m3s"],
    anomaly_threshold=4.0,
    sensor_weight=0.65,
    rolling_window=30,
    log_output_path="twin_demo_log.csv",
    sampling_rate_hz=1.0,
)

# ─────────────────────────────────────────────────────────────────
# Simple polynomial ROM: T_pred = a0 + a1*P + a2/mdot
# ─────────────────────────────────────────────────────────────────
def thermal_rom(sensor_data: dict) -> dict:
    P = sensor_data.get("power_w", 300)
    mdot = sensor_data.get("airflow_m3s", 0.5)
    CP, RHO = 1006.0, 1.2
    mdot_kg = mdot * RHO
    delta_t = P / (mdot_kg * CP) if mdot_kg > 0 else 0
    return {
        "temperature_c": 20.0 + delta_t,
        "power_w": P,
        "airflow_m3s": mdot,
    }

# ─────────────────────────────────────────────────────────────────
# Initialise with warmup data (nominal operating range)
# ─────────────────────────────────────────────────────────────────
N_WARMUP = 100
warmup = np.column_stack([
    25.0 + np.random.normal(0, 0.5, N_WARMUP),   # temperature ~25°C
    300.0 + np.random.normal(0, 10, N_WARMUP),    # power ~300W
    0.5 + np.random.normal(0, 0.02, N_WARMUP),    # airflow ~0.5 m³/s
])

pipeline = DigitalTwinPipeline(config=config, rom_evaluator=thermal_rom)
pipeline.initialize(warmup, config.sensor_channels)
print("Pipeline initialised with warmup data.")

# ─────────────────────────────────────────────────────────────────
# Run 200 time steps
# ─────────────────────────────────────────────────────────────────
N_STEPS = 200
ANOMALY_START = 150  # inject anomaly at step 150

results = []
for step in range(N_STEPS):
    t = float(step)

    # Simulate sensor readings
    if step < ANOMALY_START:
        T_sensor = 25.0 + 2 * np.sin(2 * np.pi * step / 50) + np.random.normal(0, 0.4)
        P_sensor = 300.0 + 20 * np.sin(2 * np.pi * step / 30) + np.random.normal(0, 5)
        flow     = 0.5 + 0.02 * np.cos(2 * np.pi * step / 80) + np.random.normal(0, 0.005)
    else:
        # Anomaly: temperature spike + airflow drop (blocked fan scenario)
        T_sensor = 25.0 + 15 * (step - ANOMALY_START) / 50 + np.random.normal(0, 1.0)
        P_sensor = 350.0 + np.random.normal(0, 10)
        flow     = max(0.1, 0.5 - 0.004 * (step - ANOMALY_START) + np.random.normal(0, 0.01))

    sensor_data = {
        "temperature_c": T_sensor,
        "power_w": P_sensor,
        "airflow_m3s": flow,
    }

    result = pipeline.run_step(timestamp=t, sensor_data=sensor_data)
    results.append(result)

# ─────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────
report = pipeline.generate_health_report()
print("\n── Health Report ──")
for k, v in report.items():
    if k != "recent_anomalies":
        print(f"  {k}: {v}")

print(f"\n  Recent anomalies ({len(report['recent_anomalies'])}):")
for a in report["recent_anomalies"]:
    print(f"    t={a['timestamp']:.0f}s  {a['channel']}={a['value']:.2f}  score={a['score']:.2f}")

# ─────────────────────────────────────────────────────────────────
# Feature extraction example
# ─────────────────────────────────────────────────────────────────
temps = [r["state"]["temperature_c"] for r in results]
features = FeatureExtractor.extract(temps)
print(f"\nTemperature signal features (all 200 steps):")
for k, v in features.items():
    print(f"  {k}: {v:.4f}")

# ─────────────────────────────────────────────────────────────────
# Save log
# ─────────────────────────────────────────────────────────────────
log_path = pipeline.save_log()
print(f"\nLog saved to: {log_path}")

# ─────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────
timestamps = [r["timestamp"] for r in results]
anomaly_scores = [r["anomaly_score"] for r in results]
is_anomaly = [r["is_anomaly"] for r in results]

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(timestamps, temps, label="Temperature (°C)", color="tomato")
axes[0].axvline(ANOMALY_START, color="red", linestyle="--", alpha=0.5, label="Anomaly injected")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend()
axes[0].set_title("Digital Twin State Monitor — Pipeline Demo")

axes[1].plot(timestamps, [r["state"]["power_w"] for r in results], color="steelblue", label="Power (W)")
axes[1].set_ylabel("Power (W)")
axes[1].legend()

axes[2].plot(timestamps, anomaly_scores, color="purple", label="Anomaly Score")
axes[2].axhline(config.anomaly_threshold, color="red", linestyle="--",
                label=f"Threshold = {config.anomaly_threshold}")
anomaly_ts = [t for t, a in zip(timestamps, is_anomaly) if a]
if anomaly_ts:
    axes[2].scatter(anomaly_ts, [config.anomaly_threshold + 0.5] * len(anomaly_ts),
                   color="red", zorder=5, s=20, label="Anomaly flagged")
axes[2].set_ylabel("Mahalanobis Score")
axes[2].set_xlabel("Time (s)")
axes[2].legend()

plt.tight_layout()
plt.show()
print("\nDemo complete.")
