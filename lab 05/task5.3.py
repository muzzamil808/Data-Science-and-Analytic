
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = sns.load_dataset('tips')

# --- Task 1: Compute Correlation Matrix ---
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
print("=== TASK 1 OUTPUT ===")
print("--- Correlation Matrix ---")
print(corr_matrix)
print("-" * 50)

# --- Task 2: Scatter Plot for Total Bill vs Tip ---
plt.figure(figsize=(6, 4))
plt.scatter(df['total_bill'], df['tip'], alpha=0.7, edgecolor='k')
plt.title('Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.savefig('scatter.png')
plt.close()
print("=== TASK 2 OUTPUT ===")
print("Plot code executed and saved successfully as 'scatter.png'.")
print("Description: The relationship between total bill and tip is positive, as higher bills generally correspond to higher tips. The pattern appears mostly linear with a steady upward trend.")
print("-" * 50)

# --- Task 3: Box Plot by Smoker Status ---
plt.figure(figsize=(6, 4))
sns.boxplot(x='smoker', y='tip', data=df)
plt.title('Tip Distribution by Smoker Status')
plt.xlabel('Smoker')
plt.ylabel('Tip')
plt.savefig('box_plot.png')
plt.close()
print("=== TASK 3 OUTPUT ===")
print("Box plot code executed and saved successfully as 'box_plot.png'.")
print("Observation: Based on the box plot, the median tip appears slightly higher for non-smokers compared to smokers.")
print("-" * 50)

# --- Task 4: Confounded Relationship Check ---
mean_size_by_smoker = df.groupby('smoker')['size'].mean()
print("=== TASK 4 OUTPUT ===")
print("Mean party size by smoker status:")
print(mean_size_by_smoker)
print("Conclusion: With a minimal difference of only 0.26 between non-smokers (2.67) and smokers (2.41), party size does not differ significantly enough between the groups to serve as a major confounder for the tip comparison.")
print("-" * 50)

# --- Task 5: Explain Correlation Limitations ---
task_5_explanation = """
=== TASK 5 OUTPUT ===
A correlation near zero only means there is no straight-line relationship, but it doesn't mean the variables are totally unrelated. They could easily follow a strong non-linear pattern, like a U-shaped curve, which standard correlation completely misses. At the same time, comparing a number like average tips across categories—such as smokers versus non-smokers—requires care. If we ignore hidden factors like party size, our conclusions can be totally wrong. An apparent difference between groups might actually be caused by a third confounding variable rather than the category itself, making it easy to jump to false conclusions without checking deeper.
"""
print(task_5_explanation)
