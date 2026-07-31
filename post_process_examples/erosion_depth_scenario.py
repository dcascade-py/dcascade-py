import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# 1. OUTPUT DIRECTORY
# ==================================================
output_dir = r"D:/0Padova Study/sem 4/GitHub/dcascade-py/cascade_results"
os.makedirs(output_dir, exist_ok=True)

# ==================================================
# 2. FILE PATHS FOR 4 CASES
# ==================================================
cases = [
    {
        "label": "AL0.2_50bf",
        "result_file": r"D:/0Padova Study/sem 4/GitHub/dcascade-py/cascade_results/Tagliamento_50bf_al002_ext.p",
        "network_file": r"D:/0Padova Study/sem 4/GitHub/dcascade-py/inputs/Tagliamento_river/Reach_data_tag_50bf.csv",
        "threshold": 0.2
    },
    {
        "label": "AL0.5_50bf",
        "result_file": r"D:/0Padova Study/sem 4/GitHub/dcascade-py/cascade_results/Tagliamento_50bf_al005_ext.p",
        "network_file": r"D:/0Padova Study/sem 4/GitHub/dcascade-py/inputs/Tagliamento_river/Reach_data_tag_50bf.csv",
        "threshold": 0.5
    },
    {
        "label": "AL0.5_bf",
        "result_file": r"D:/0Padova Study/sem 4/GitHub/dcascade-py/cascade_results/Tagliamento_bf_al005_ext.p",
        "network_file": r"D:/0Padova Study/sem 4/GitHub/dcascade-py/inputs/Tagliamento_river/Reach_data_tag_bf.csv",
        "threshold": 0.5
    },
    {
        "label": "AL0.2_bf",
        "result_file": r"D:/0Padova Study/sem 4/GitHub/dcascade-py/cascade_results/Tagliamento_bf_al002_ext.p",
        "network_file": r"D:/0Padova Study/sem 4/GitHub/dcascade-py/inputs/Tagliamento_river/Reach_data_tag_bf.csv",
        "threshold": 0.2
    },
]

# ==================================================
# 3. FUNCTION TO PROCESS ONE CASE
# ==================================================
def process_case(case):
    result_file = case["result_file"]
    network_file = case["network_file"]
    threshold = case["threshold"]
    label = case["label"]

    # -------------------------------
    # Load output pickle
    # -------------------------------
    with open(result_file, "rb") as f:
        extended_output = pickle.load(f)

    # -------------------------------
    # Extract volume out
    # shape: (time, reach, class)
    # -------------------------------
    vol_out_class = np.asarray(extended_output["Volume out per grain sizes [m^3]"])
    vol_out = vol_out_class.sum(axis=2)   # shape: (time, reach)

    print(f"{label} Volume out shape: {vol_out.shape}")

    # -------------------------------
    # Extract widths
    # -------------------------------
    widths = np.asarray(extended_output["Widths [m]"])
    print(f"{label} Widths shape: {widths.shape}")

    # -------------------------------
    # Load reach lengths
    # -------------------------------
    reach_df = pd.read_csv(network_file)

    possible_length_cols = ["Length", "length", "LENGTH"]
    length_col = None
    for c in possible_length_cols:
        if c in reach_df.columns:
            length_col = c
            break

    if length_col is None:
        raise ValueError(
            f"[{label}] No length column found. Available columns: {list(reach_df.columns)}"
        )

    lengths = reach_df[length_col].to_numpy(dtype=float)

    if len(lengths) != vol_out.shape[1]:
        raise ValueError(
            f"[{label}] Length count ({len(lengths)}) does not match "
            f"number of reaches ({vol_out.shape[1]})."
        )

    lengths_2d = lengths[np.newaxis, :]   # shape: (1, reach)

    # -------------------------------
    # Compute area and depth
    # -------------------------------
    area = widths * lengths_2d

    if np.any(area <= 0):
        raise ValueError(f"[{label}] Some area values are zero or negative")

    depth = vol_out / area

    # -------------------------------
    # Threshold analysis
    # -------------------------------
    exceed = depth > threshold

    count_exceed_per_reach = exceed.sum(axis=0)
    count_exceed_per_timestep = exceed.sum(axis=1)

    total_depth_per_reach = depth.sum(axis=0)
    max_depth_per_reach = depth.max(axis=0)

    # optional extra info
    mean_depth_per_reach = depth.mean(axis=0)

    # -------------------------------
    # Build labels
    # -------------------------------
    n_timesteps, n_reaches = depth.shape
    reach_names = [f"reach_{i+1}" for i in range(n_reaches)]
    time_names = [f"t{i}" for i in range(n_timesteps)]

    # -------------------------------
    # Build dataframes
    # -------------------------------
    depth_df = pd.DataFrame(depth, index=time_names, columns=reach_names)
    exceed_df = pd.DataFrame(exceed, index=time_names, columns=reach_names)

    summary_reach_df = pd.DataFrame({
        "reach": reach_names,
        "total_depth_m": total_depth_per_reach,
        "mean_depth_m": mean_depth_per_reach,
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
                    "case": label,
                    "timestep": time_names[t],
                    "reach": reach_names[r],
                    "depth_m": depth[t, r]
                })

    events_df = pd.DataFrame(events)

    return {
        "label": label,
        "threshold": threshold,
        "reach_names": reach_names,
        "depth_df": depth_df,
        "exceed_df": exceed_df,
        "summary_reach_df": summary_reach_df,
        "summary_time_df": summary_time_df,
        "events_df": events_df,
        "count_exceed_per_reach": count_exceed_per_reach,
        "count_exceed_per_timestep": count_exceed_per_timestep,
        "total_depth_per_reach": total_depth_per_reach,
        "max_depth_per_reach": max_depth_per_reach,
        "mean_depth_per_reach": mean_depth_per_reach,
    }

# ==================================================
# 4. PROCESS ALL CASES
# ==================================================
results = []
for case in cases:
    print(f"Processing {case['label']} ...")
    result = process_case(case)
    results.append(result)

# ==================================================
# 5. EXPORT ALL RESULTS TO ONE EXCEL FILE
# ==================================================
output_excel = os.path.join(output_dir, "volume_out_depth_analysis_4cases.xlsx")

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    for result in results:
        label = result["label"]

        result["depth_df"].to_excel(writer, sheet_name=f"{label}_depth")
        result["exceed_df"].to_excel(writer, sheet_name=f"{label}_exceed")
        result["summary_reach_df"].to_excel(writer, sheet_name=f"{label}_reachsum", index=False)
        result["summary_time_df"].to_excel(writer, sheet_name=f"{label}_timesum", index=False)

        if not result["events_df"].empty:
            result["events_df"].to_excel(writer, sheet_name=f"{label}_events", index=False)

print(f"Excel file saved as: {output_excel}")

# ==================================================
# 6. COMBINED PLOT 1
#    Threshold exceedance count per reach
# ==================================================
plt.figure(figsize=(14, 7))

for result in results:
    x = np.arange(len(result["reach_names"]))
    plt.plot(
        x,
        result["count_exceed_per_reach"],
        marker="o",
        label=f"{result['label']} (thr={result['threshold']})"
    )

plt.xlabel("Reach")
plt.ylabel("Number of times depth exceeded threshold")
plt.title("Threshold exceedance count per reach for 4 cases")
plt.xticks(np.arange(len(results[0]["reach_names"])), results[0]["reach_names"], rotation=90)
plt.legend()
plt.tight_layout()

plot1_file = os.path.join(output_dir, "combined_reach_vs_threshold_crossings_4cases.png")
plt.savefig(plot1_file, dpi=300, bbox_inches="tight")
plt.close()

# ==================================================
# 7. COMBINED PLOT 2
#    Total erosion depth per reach
# ==================================================
plt.figure(figsize=(14, 7))

for result in results:
    x = np.arange(len(result["reach_names"]))
    plt.plot(
        x,
        result["total_depth_per_reach"],
        marker="o",
        label=result["label"]
    )

plt.xlabel("Reach")
plt.ylabel("Total depth of erosion [m]")
plt.title("Total depth of erosion per reach for 4 cases")
plt.xticks(np.arange(len(results[0]["reach_names"])), results[0]["reach_names"], rotation=90)
plt.legend()
plt.tight_layout()

plot2_file = os.path.join(output_dir, "combined_reach_vs_total_erosion_depth_4cases.png")
plt.savefig(plot2_file, dpi=300, bbox_inches="tight")
plt.close()

# ==================================================
# 8. OPTIONAL COMBINED PLOT 3
#    Maximum single timestep depth per reach
# ==================================================
plt.figure(figsize=(14, 7))

for result in results:
    x = np.arange(len(result["reach_names"]))
    plt.plot(
        x,
        result["max_depth_per_reach"],
        marker="o",
        label=result["label"]
    )

plt.xlabel("Reach")
plt.ylabel("Maximum depth in a single timestep [m]")
plt.title("Maximum single timestep depth per reach for 4 cases")
plt.xticks(np.arange(len(results[0]["reach_names"])), results[0]["reach_names"], rotation=90)
plt.legend()
plt.tight_layout()

plot3_file = os.path.join(output_dir, "combined_reach_vs_max_depth_4cases.png")
plt.savefig(plot3_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot 1 saved as: {plot1_file}")
print(f"Plot 2 saved as: {plot2_file}")
print(f"Plot 3 saved as: {plot3_file}")