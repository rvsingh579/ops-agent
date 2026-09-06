import pandas as pd
import matplotlib.pyplot as plt

rawdf = pd.read_csv("/Users/raviranjan/Documents/mera_kaam/ops-agent/data/raw/train_FD001.txt", sep="\s", header=None)

n_sensors = len(rawdf.columns) - 5

rawdf.columns = (
    ["unit_number", "time_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, n_sensors + 1)]
)

df = rawdf[rawdf["unit_number"] == 3]
print(max(df["time_cycles"]))

sensor_std = (
    rawdf[[f"sensor_{i}" for i in range(1, n_sensors + 1)]]
    .std()
    .sort_values()
)

print(sensor_std)

threshold = 1e-3

low_variance_sensors = sensor_std[sensor_std < threshold]
non_trivial_sensors = sensor_std[sensor_std >= threshold]

print("=== NEAR-ZERO STD ===")
print(low_variance_sensors)

print("\n=== NON-TRIVIAL STD ===")
print(non_trivial_sensors)

min_sensor = low_variance_sensors.idxmin()
max_sensor = non_trivial_sensors.idxmax()

df.sort_values(by="time_cycles", inplace=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(df["time_cycles"], df[min_sensor])
axes[0].set_title(f"Sensor with lowest std: {min_sensor}")
axes[0].set_xlabel("Time Cycles")
axes[0].set_ylabel("Sensor Value")

axes[1].plot(df["time_cycles"], df[max_sensor])
axes[1].set_title(f"Sensor with highest std: {max_sensor}")
axes[1].set_xlabel("Time Cycles")
axes[1].set_ylabel("Sensor Value")

plt.tight_layout()
plt.show()