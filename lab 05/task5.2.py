# =====================================================================
# ALL TASKS COMBINED: OUTLIER DETECTION & STATISTICAL ANALYSIS LAB
# =====================================================================

# --- Task 1: Generate Outlier Detection Code (IQR Rule) ---
import numpy as np

data = np.array([10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 10, 13, 12, 13, 10, 12, 11, 14, 13, 15])

q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]

print("=== TASK 1 OUTPUT ===")
print(f"Q1: {q1:.2f}")
print(f"Q3: {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Lower Bound: {lower_bound:.2f}")
print(f"Upper Bound: {upper_bound:.2f}")
print("Identified Outliers (IQR):", list(outliers_iqr))
print("-" * 50)


# --- Task 2: Z-Score Outlier Calculation ---
mean_val = np.mean(data)
std_val = np.std(data, ddof=0)

z_scores = (data - mean_val) / std_val
outliers_z = data[np.abs(z_scores) > 3]

print("=== TASK 2 OUTPUT ===")
print(f"Mean: {mean_val:.4f}")
print(f"Standard Deviation: {std_val:.4f}")
print("Identified Outliers (Z-score > 3):", list(outliers_z))
print("-" * 50)


# --- Task 3: Histogram Shape Interpretation ---
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.hist(data, bins=10, edgecolor='black')
plt.title('Histogram of Dataset with Outlier')
plt.xlabel('Value')
plt.ylabel('Frequency')
# plt.show() # Uncomment to display plot interactively

print("=== TASK 3 OUTPUT ===")
print("Plot code executed successfully.")
print("Description: The distribution is heavily right-skewed. The extreme value 102 stretches the right tail, causing the mean to be significantly greater than the median because the outlier pulls the mean upward while the median remains stable.")
print("-" * 50)


# --- Task 4: IQR vs Z-Score Comparison ---
skewed_data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5, 50])

# IQR for skewed data
q1_s = np.percentile(skewed_data, 25)
q3_s = np.percentile(skewed_data, 75)
iqr_s = q3_s - q1_s
lower_s = q1_s - 1.5 * iqr_s
upper_s = q3_s + 1.5 * iqr_s
iqr_outliers = skewed_data[(skewed_data < lower_s) | (skewed_data > upper_s)]

# Z-score for skewed data
mean_s = np.mean(skewed_data)
std_s = np.std(skewed_data, ddof=0)
z_s = (skewed_data - mean_s) / std_s
z_outliers = skewed_data[np.abs(z_s) > 3]

print("=== TASK 4 OUTPUT ===")
print("Skewed Data IQR Outliers:", list(iqr_outliers))
print("Z-scores for all points:", np.round(z_s, 2))
print("Skewed Data Z-Score Outliers (|z| > 3):", list(z_outliers))

"""
Comment on Z-Score behavior in skewed datasets:
The Z-score method relies on the mean and standard deviation, which are heavily distorted by extreme values. 
In this dataset, the outlier 50 inflates the standard deviation (~14.14) and pulls the mean up (7.7), resulting in a z-score of ~2.99 for 50. 
Because 2.99 is just below the strict threshold of 3, the Z-score method fails to flag the obvious outlier 50, whereas the robust IQR method successfully identifies it.
"""
print("-" * 50)
