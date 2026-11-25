import matplotlib.pyplot as plt

# --- Task 1: Self-Explanatory Line Chart ---
x1 = [1, 2, 3, 4, 5]
y1 = [10, 20, 15, 30, 25]

fig1, ax1 = plt.subplots()
ax1.plot(x1, y1, label='Actuals')
ax1.set_title('Monthly Sales Trend')
ax1.set_xlabel('Month')
ax1.set_ylabel('Revenue (USD)')
ax1.legend()

# --- Task 2: Accessible Colour and Style ---
x_vals = [1, 2, 3]
y_a = [1, 2, 3]
y_b = [3, 2, 1]

fig2, ax2 = plt.subplots()
ax2.plot(x_vals, y_a, color='blue', linestyle='-', marker='o', label='Group A')
ax2.plot(x_vals, y_b, color='orange', linestyle='--', marker='s', label='Group B')
ax2.legend()

# --- Task 3: Honest Bar Chart Axis ---
categories = ['A', 'B', 'C']
values = [10, 11, 12]

fig3, ax3 = plt.subplots()
ax3.bar(categories, values)
ax3.set_ylim(bottom=0)
ax3.set_title('Comparison of Metrics')
ax3.set_ylabel('Value')

# --- Task 4: Annotating Key Data Point ---
x_scatter = [1, 2, 3, 4]
y_scatter = [5, 10, 8, 15]

fig4, ax4 = plt.subplots()
ax4.scatter(x_scatter, y_scatter)
ax4.set_title('Data Distribution')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.annotate(
    'Peak Value',
    xy=(4, 15),
    xytext=(3, 13),
    arrowprops=dict(facecolor='black', shrink=0.05),
)

plt.show()