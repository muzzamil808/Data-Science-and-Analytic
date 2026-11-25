import matplotlib.pyplot as plt
import numpy as np

# Task 1
fig1, ax1 = plt.subplots()
ax1.plot([1, 2, 3], [2, 4, 6])
print("--- Task 1 ---")
print(type(fig1))
print(type(ax1))
print()

# Task 2
months = [1, 2, 3, 4, 5, 6]
sales = [10, 12, 11, 15, 18, 20]
fig2, ax2 = plt.subplots()
ax2.plot(months, sales)
fig2.savefig('sales_trend.png', dpi=150, bbox_inches='tight')
print("--- Task 2 ---")
print("File 'sales_trend.png' successfully saved with dpi=150 and bbox_inches='tight'.")
print()

# Task 3
np.random.seed(42)
scores = np.random.randint(50, 100, 20)
fig3, ax3 = plt.subplots()
ax3.hist(scores, bins=5)
print("--- Task 3 ---")
print(ax3.patches)
print()

# Task 4
study_hours = [1, 2, 3, 4, 5]
test_scores = [50, 60, 70, 85, 95]
fig4, ax4 = plt.subplots()
ax4.scatter(study_hours, test_scores)
print("--- Task 4 ---")
print(ax4.collections)