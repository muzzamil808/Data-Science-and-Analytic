import pandas as pd
import numpy as np

# Task 1
data1 = {'ID': [1, 2, 2, 3, 4, 4], 'Name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David']}
df1 = pd.DataFrame(data1)
print(df1)
print(df1.duplicated().sum())
df1_cleaned = df1.drop_duplicates()
print(df1_cleaned)

# Task 2
data2 = {'Score': [80, np.nan, 90, 70, np.nan, 100]}
df2 = pd.DataFrame(data2)
print(df2['Score'])
median_val = df2['Score'].median()
df2['Score'] = df2['Score'].fillna(median_val)
print(df2['Score'])

# Task 3
data3 = {'Product': ['Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E'], 'Price': ['45.50', '12.a', '100.0', 'Unknown', '25.75']}
df3 = pd.DataFrame(data3)
print(df3)
print(df3.dtypes)
df3['Price'] = pd.to_numeric(df3['Price'], errors='coerce')
print(df3)
print(df3.dtypes)

# Task 4
data4 = {'Customer': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'], 'City': [' London', 'london', 'LONDON', 'New York ', 'new york']}
df4 = pd.DataFrame(data4)
print(df4['City'].unique())
df4['City_Cleaned'] = df4['City'].str.strip().str.lower()
print(df4['City_Cleaned'].unique())
print(df4)