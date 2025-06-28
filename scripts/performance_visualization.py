#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

# load monitoring metrics
monitor_path = "datamart/gold/monitoring/metrics_2025-06-01_to_2025-06-28.parquet"
df = pd.read_parquet(monitor_path)
df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

# ensure output dir
out_dir = "datamart/gold/monitoring/plots"
os.makedirs(out_dir, exist_ok=True)

# Plot AUC over time
plt.figure()
plt.plot(df["snapshot_date"], df["test_auc"], marker="o")
plt.title("Model Test AUC Over Time")
plt.xlabel("Snapshot Date")
plt.ylabel("Test AUC")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "auc_over_time.png"))
print(f"Saved plot to {out_dir}/auc_over_time.png")