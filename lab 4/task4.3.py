import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Sample DataFrame setup for demonstration
np.random.seed(42)
df = pd.DataFrame({
    'batch': ['A', 'B', 'A', 'B', 'C', 'C', 'A', 'B', 'C', 'A'],
    'score': np.random.randint(60, 100, 10),
    'group': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
    'value': np.random.randint(10, 50, 10),
    'var1': np.random.rand(10),
    'var2': np.random.rand(10)
})

# --- Task 1: Generate Box Plot Code ---
fig1, ax1 = plt.subplots()
sns.boxplot(data=df, x='batch', y='score', ax=ax1)
plt.title('Task 1: Box Plot')
plt.show()

# --- Task 2: Create Correlation Heatmap ---
fig2, ax2 = plt.subplots()
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
plt.title('Task 2: Correlation Heatmap')
plt.show()

# --- Task 3: Pair Plot for EDA ---
sns.pairplot(df.select_dtypes(include=[np.number]))
plt.show()

# --- Task 4: Matplotlib Fallback for Control ---
fig4, ax4 = plt.subplots()
ax = sns.boxplot(data=df, x='group', y='value', ax=ax4)
ax.set_title('Custom Title')
plt.show()