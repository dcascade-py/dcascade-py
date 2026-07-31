import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------
result_file = r"../cascade_results/Tagliamento_50bf_al005_ext.p"
network_file = r"D:/0Padova Study/sem 4/GitHub/dcascade-py/inputs/Tagliamento_river/Reach_data_tag_50bf.csv"
threshold = 0.5

# --------------------------------------------------
# 2. LOAD OUTPUT
# --------------------------------------------------
with open(result_file, "rb") as f:
    extended_output = pickle.load(f)

# --------------------------------------------------
# 3. EXTRACT VOLUME OUT
#    shape: (time, reach, class)
# --------------------------------------------------
vol_out_class = np.asarray(extended_output["Volume out per grain sizes [m^3]"])
vol_out = vol_out_class.sum(axis=2)   # shape: (time, reach)

print("Volume out shape:", vol_out.shape)

# --------------------------------------------------
# 4. EXTRACT WIDTHS
# --------------------------------------------------
widths = np.asarray(extended_output["Widths [m]"])
print("Widths shape:", widths.shape)

# --------------------------------------------------
# 5. LOAD LENGTHS
# --------------------------------------------------
reach_df = pd.read_csv(network_file)

possible_length_cols = ["Length", "length", "LENGTH"]
length_col = None
for c in possible_length_cols:
    if c in reach_df.columns:
        length_col = c
        break

if length_col is None:
    raise ValueError(f"No length column found. Available columns: {list(reach_df.columns)}")

lengths = reach_df[length_col].to_numpy(dtype=float)

if len(lengths) != vol_out.shape[1]:
    raise ValueError(
        f"Length count ({len(lengths)}) does not match number of reaches ({vol_out.shape[1]})."
    )

lengths_2d = lengths[np.newaxis, :]   # shape (1, reach)

# --------------------------------------------------
# 6. COMPUTE AREA AND DEPTH
# --------------------------------------------------
area = widths * lengths_2d   # shape: (time, reach)

if np.any(area <= 0):
    raise ValueError("Some area values are zero or negative")

depth = vol_out / area       # shape: (time, reach)

# --------------------------------------------------
# 7. CHECK THRESHOLD
# --------------------------------------------------
exceed = depth > threshold

count_exceed_per_reach = exceed.sum(axis=0)
count_exceed_per_timestep = exceed.sum(axis=1)

total_depth_per_reach = depth.sum(axis=0)
max_depth_per_reach = depth.max(axis=0)

# --------------------------------------------------
# 8. BUILD TABLES
# --------------------------------------------------
n_timesteps, n_reaches = depth.shape

reach_names = [f"reach_{i+1}" for i in range(n_reaches)]
time_names = [f"t{i}" for i in range(n_timesteps)]

depth_df = pd.DataFrame(depth, index=time_names, columns=reach_names)
exceed_df = pd.DataFrame(exceed, index=time_names, columns=reach_names)

summary_reach_df = pd.DataFrame({
    "reach": reach_names,
    "total_depth_m": total_depth_per_reach,
    f"count_depth_gt_{threshold}m": count_exceed_per_reach,
    "max_depth_single_timestep_m": max_depth_per_reach
})

summary_time_df = pd.DataFrame({
    "timestep": time_names,
    f"n_reaches_depth_gt_{threshold}m": count_exceed_per_timestep
})

events = []
for t in range(n_timesteps):
    for r in range(n_reaches):
        if exceed[t, r]:
            events.append({
                "timestep": time_names[t],
                "reach": reach_names[r],
                "depth_m": depth[t, r]
            })

events_df = pd.DataFrame(events)

# --------------------------------------------------
# 9. EXPORT TO EXCEL
# --------------------------------------------------
output_excel = "volume_out_depth_analysis.xlsx"

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    depth_df.to_excel(writer, sheet_name="Depth_per_timestep")
    exceed_df.to_excel(writer, sheet_name="Threshold_exceedance")
    summary_reach_df.to_excel(writer, sheet_name="Summary_per_reach", index=False)
    summary_time_df.to_excel(writer, sheet_name="Summary_per_timestep", index=False)
    events_df.to_excel(writer, sheet_name="Exceedance_events", index=False)

print(f"Excel file saved as: {output_excel}")

# --------------------------------------------------
# 10. SAVE PLOTS AS PNG
# --------------------------------------------------

# Plot 1: reach vs number of times threshold is crossed
plt.figure(figsize=(12, 6))
plt.bar(reach_names, count_exceed_per_reach)
plt.xlabel("Reach")
plt.ylabel(f"Number of times depth > {threshold} m")
plt.title(f"Threshold exceedance count per reach (threshold = {threshold} m)")
plt.xticks(rotation=90)
plt.tight_layout()
plot1_file = "reach_vs_threshold_crossings.png"
plt.savefig(plot1_file, dpi=300, bbox_inches="tight")
plt.close()

# Plot 2: reach vs total depth of erosion
plt.figure(figsize=(12, 6))
plt.bar(reach_names, total_depth_per_reach)
plt.xlabel("Reach")
plt.ylabel("Total depth of erosion [m]")
plt.title("Total depth of erosion per reach")
plt.xticks(rotation=90)
plt.tight_layout()
plot2_file = "reach_vs_total_erosion_depth.png"
plt.savefig(plot2_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot 1 saved as: {plot1_file}")
print(f"Plot 2 saved as: {plot2_file}")