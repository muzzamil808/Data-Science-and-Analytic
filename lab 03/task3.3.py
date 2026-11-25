import pandas as pd
import numpy as np

# Task 1
sales = pd.DataFrame({
    'region': ['North', 'South', 'North', 'East', 'South'],
    'amount': [100, 200, 150, 50, 250]
})
result_task1 = sales.groupby('region')['amount'].mean()
print("--- Task 1 ---")
print(result_task1)
print()

# Task 2
result_task2 = sales.groupby('region')['amount'].agg(['min', 'max', 'sum'])
print("--- Task 2 ---")
print(result_task2)
print()

# Task 3
employees = pd.DataFrame({
    'department': ['HR', 'IT', 'HR', 'IT'],
    'year': [2023, 2023, 2024, 2024],
    'salary': [50000, 60000, 55000, 65000]
})
pivot_task3 = pd.pivot_table(employees, values='salary', index='department', columns='year', aggfunc='mean')
print("--- Task 3 ---")
print(pivot_task3)
print()

# Task 4
orders = pd.DataFrame({
    'order_id': [1, 2, 3],
    'product': ['A', 'B', 'C']
})
customers = pd.DataFrame({
    'order_id': [1, 2, 4],
    'name': ['Alice', 'Bob', 'Charlie']
})
inner_merge = pd.merge(orders, customers, on='order_id', how='inner')
left_merge = pd.merge(orders, customers, on='order_id', how='left')

print("--- Task 4 (Inner Merge) ---")
print(inner_merge)
print("\n--- Task 4 (Left Merge) ---")
print(left_merge)
print()

# Task 5 (Explanation)
# The inner merge only keeps rows that have a match in both tables, which is why 
# order 3 gets dropped since it lacks a customer record. A left merge, however, 
# keeps everything from the orders table and just fills in NaN where the customer 
# data is missing. This matters a lot with real datasets because an inner merge 
# would quietly drop incomplete records and make you undercount total orders, 
# while a left merge keeps all your transactions visible so you can catch and 
# handle missing data properly.