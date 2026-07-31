# -*- coding: utf-8 -*-
"""
Plot fraction of volume out per grain size along reach index.

Legend is placed outside the plot on the right side to avoid overlap.
"""

# =========================================================
# LIBRARIES
# =========================================================
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# =========================================================
# PATHS
# =========================================================
path = "..\\cascade_results\\"

# Change these depending on the scenario you want to plot
name_simu = "Tagliamento_50bf_al005"
name_simu_ext = "Tagliamento_50bf_al005_ext"

# Folder to store the plots
figure_folder = os.path.join(path, f"fraction_GS_Vout_reaches_sum_{name_simu}")

if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)


# =========================================================
# LOAD BASIC OUTPUT
# =========================================================
with open(os.path.join(path, name_simu + ".p"), "rb") as f:
    data_output = pickle.load(f)

psi = data_output["Simulation parameters"]["psi"]

n_class = len(psi)

# Grain size classes in mm
dmi = 2 ** (-psi)
dmi = np.squeeze(dmi)


# =========================================================
# LOAD EXTENDED OUTPUT
# =========================================================
with open(os.path.join(path, name_simu_ext + ".p"), "rb") as f:
    data_output_ext = pickle.load(f)

# Shape expected:
# time x reach x grain_size_class
Qbi_mob = data_output_ext["Volume out per grain sizes [m^3]"]


# =========================================================
# CALCULATE FRACTION PER GRAIN SIZE
# =========================================================
# Sum over all timesteps
# Result shape: reach x grain_size_class
vout_tot = np.sum(Qbi_mob, axis=0)

# Total per reach
reach_total = np.sum(vout_tot, axis=1, keepdims=True)

# Avoid division by zero
reach_total[reach_total == 0] = np.nan

# Fraction per grain size
Fir_vout = vout_tot / reach_total

# Replace NaN with 0 where there was no transported sediment
Fir_vout = np.nan_to_num(Fir_vout, nan=0.0)

n_reach, n_categories = Fir_vout.shape


# =========================================================
# COLORS
# =========================================================
colors = [plt.cm.jet(i / (n_categories - 1)) for i in range(n_categories)]


# =========================================================
# PLOT STACKED BAR CHART
# =========================================================
fig, ax = plt.subplots(figsize=(18, 6.5))

bottom = np.zeros(n_reach)

# X-axis as reach index / FromN
reach_FromN = np.arange(1, n_reach + 1, 1)

for i in range(n_categories):
    ax.bar(
        reach_FromN,
        Fir_vout[:, i],
        bottom=bottom,
        color=colors[i],
        width=0.8,
        label=f"d = {dmi[i]} mm"
    )

    bottom += Fir_vout[:, i]


# =========================================================
# AXIS FORMATTING
# =========================================================
ax.set_xlabel("Reach index (FromN)", fontsize=18)
ax.set_ylabel("Volume fraction per grain size", fontsize=16)

ax.tick_params(axis="y", which="major", labelsize=15)
ax.tick_params(axis="x", which="major", labelsize=12)

ax.set_ylim(0, 1.0)
ax.set_xlim(0, n_reach + 1)

ax.grid(axis="y", alpha=0.3)


# =========================================================
# LEGEND OUTSIDE ON THE RIGHT
# =========================================================
ax.legend(
    fontsize=10,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.0,
    frameon=True,
    title="Grain size"
)

# Leave space on the right for the legend
fig.subplots_adjust(right=0.72)


# =========================================================
# SAVE FIGURE
# =========================================================
figure_path_png = os.path.join(
    figure_folder,
    "fraction_per_GS_Vout_legend_right.png"
)

figure_path_pdf = os.path.join(
    figure_folder,
    "fraction_per_GS_Vout_legend_right.pdf"
)

fig.savefig(
    figure_path_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    figure_path_pdf,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =========================================================
# SAVE CSV
# =========================================================
csv_path = os.path.join(
    figure_folder,
    "fraction_per_GS_Vout.csv"
)

df_fraction = pd.DataFrame({
    "Reach_FromN": reach_FromN
})

for i in range(n_categories):
    df_fraction[f"d_{dmi[i]}_mm"] = Fir_vout[:, i]

df_fraction.to_csv(csv_path, index=False)


# =========================================================
# DONE
# =========================================================
print("Done.")
print("Figure saved:")
print(figure_path_png)
print(figure_path_pdf)
print("CSV saved:")
print(csv_path)