# --- Task 1: Load and Inspect Tips Dataset ---
import seaborn as sns

df = sns.load_dataset('tips')

print("=== TASK 1 OUTPUT ===")
print("--- df.info() ---")
df.info()

print("\n--- df.head() ---")
print(df.head())

numeric_cols = list(df.select_dtypes(include=['number']).columns)
categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)
print("-" * 50)


# --- Task 2: Compute Centre and Spread ---
tb = df['total_bill']

mean_val = tb.mean()
median_val = tb.median()
std_val = tb.std()
skew_val = tb.skew()
q1 = tb.quantile(0.25)
q3 = tb.quantile(0.75)
iqr = q3 - q1

print("=== TASK 2 OUTPUT ===")
print(f"Mean: {mean_val:.4f}")
print(f"Median: {median_val:.4f}")
print(f"Standard Deviation: {std_val:.4f}")
print(f"Skewness: {skew_val:.4f}")
print(f"Q1: {q1:.4f}")
print(f"Q3: {q3:.4f}")
print(f"IQR: {iqr:.4f}")
print("-" * 50)


# --- Task 3: Categorical Distribution Summary ---
print("=== TASK 3 OUTPUT ===")
print("--- Day Value Counts ---")
print(df['day'].value_counts())

print("\n--- Sex Value Counts ---")
print(df['sex'].value_counts())

print("Total row count:", len(df))
print("Sum of sex value counts:", df['sex'].value_counts().sum())
print("Summary: Sat is most frequent (87), Fri is least frequent (19). Sex column has no missing values.")
print("-" * 50)


# --- Task 4: Grouped Descriptive Statistics ---
grouped = df.groupby('smoker')['total_bill'].agg(['mean', 'median'])

print("=== TASK 4 OUTPUT ===")
print(grouped)
print("Justification: The median is more appropriate for describing typical bill size due to right-skewness and outliers.")