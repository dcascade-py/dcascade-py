# -*- coding: utf-8 -*-
"""
Linear regression per reach using Excel file
Input columns required:
- FromN
- distance
- Z
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# INPUT FILE
# --------------------------------------------------
profile = pd.read_excel(
    r"D:\0Padova Study\sem 4\GitHub\dcascade-py\inputs\Tagliamento_river\for regression.xlsx"
)

# Clean column names in case Excel added spaces
profile.columns = profile.columns.str.strip()

print("Columns in file:", profile.columns.tolist())

# --------------------------------------------------
# OUTPUT FOLDER
# --------------------------------------------------
output_folder = r"D:\0Padova Study\sem 4\GitHub\dcascade-py\inputs\Tagliamento_river\Results_regression"
os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------------
# LIST OF REACHES
# --------------------------------------------------
FromN_list = sorted(profile["FromN"].dropna().unique())

# --------------------------------------------------
# REGRESSION PER REACH
# --------------------------------------------------
store_results = []

for FromN in FromN_list:
    mydata = profile.loc[profile["FromN"] == FromN].copy()
    mydata = mydata.sort_values(by="distance")

    if len(mydata) < 2:
        continue

    fig = plt.figure()
    ax = plt.subplot(111)

    # Distance in meters
    dist = mydata["distance"]
    elev = mydata["Z"]

    # Plot original elevation profile
    ax.plot(dist, elev, linewidth=1.8, label="Elevation profile")

    # Linear regression
    X = dist.values.reshape((-1, 1))
    Y = elev.values

    model = LinearRegression().fit(X, Y)
    r_sq = model.score(X, Y)
    intercept = model.intercept_
    slope = model.coef_[0]

    # Regression line
    X2 = dist.values
    Y2 = intercept + slope * X2

    # Formula text
    eq_text = f"y = {slope:.8f}x + {intercept:.6f}\nR² = {r_sq:.8f}"

    ax.plot(
        X2,
        Y2,
        '--',
        linewidth=3,
        label=eq_text
    )

    # Figure formatting
    ax.set_xlabel("Distance (m)", fontsize=18)
    ax.set_ylabel("Elevation (m)", fontsize=18)
    ax.set_title(f"FromN: {FromN}", fontsize=20)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=11, loc="best")

    fig.set_tight_layout(True)
    fig.set_size_inches(10, 8)

    # Save figure
    fig.savefig(
        os.path.join(output_folder, f"FromN_{FromN}_regression.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Store output values
    store_results.append({
        "FromN": FromN,
        "Slope": slope,
        "Intercept": intercept,
        "RSQ": r_sq
    })

# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------
results_df = pd.DataFrame(store_results)

results_df.to_csv(
    os.path.join(output_folder, "all_reaches_regression_results.csv"),
    index=False
)

print("Done.")
print(f"Plots saved in: {output_folder}")
print(f"CSV saved as: {os.path.join(output_folder, 'all_reaches_regression_results.csv')}")