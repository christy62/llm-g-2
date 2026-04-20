import pandas as pd
import matplotlib.pyplot as plt

df_theta = pd.read_csv("/home/christy/helix_generator/eic/theta_table.csv")

# Only required PIDs
allowed_pids = [11, 211, 321, 2212]
df_theta = df_theta[df_theta["PID"].isin(allowed_pids)]

# PID labels and colors
particles = {
    11: ("Electron", "red"),
    211: ("Pion", "blue"),
    321: ("Kaon", "green"),
    2212: ("Proton", "purple")
}

plt.figure(figsize=(10,6))

for pid, (name, color) in particles.items():
    subset = df_theta[df_theta["PID"] == pid]

    plt.scatter(
        subset["mom"],
        subset["mean_theta_c_aero"],
        label=name,
        color=color,
        alpha=0.7
    )

plt.xlabel("Momentum [GeV/c]")
plt.ylabel("Mean Cherenkov Angle (Aerogel) [mrad]")
plt.title("Momentum vs Mean Cherenkov Angle (Aerogel)")

plt.xlim(0, 60)
plt.ylim(200, 230)

plt.legend()
plt.grid(True)

plt.savefig("mom_vs_mean_theta_c_aero.png", dpi=200)
print("Saved plot to mom_vs_mean_theta_c_aero.png")