import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Task 1: Reproducible Split Setup ---
np.random.seed(42)
X = np.random.rand(100, 3)
y = np.dot(X, [1.5, -2.0, 3.5]) + np.random.normal(0, 0.1, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("--- Task 1 Output ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print()

# --- Task 2: Fit and Predict API ---
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("--- Task 2 Output ---")
print("First 3 Predicted Values:", predictions[:3])
print("First 3 Actual Values   :", y_test[:3])
print()

# --- Task 3: Leak-Free Pipeline Construction ---
# Using a Pipeline prevents data leakage because preprocessing transformations 
# (like mean/variance scaling) are computed exclusively on the training split 
# during fit, ensuring validation statistics never leak into training.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)
pipeline_preds = pipeline.predict(X_test)

print("--- Task 3 Output ---")
print("Pipeline successfully fitted and predicted.")
print()

# --- Task 4: Coefficient Interpretation Task ---
coefficients = pipeline.named_steps['regressor'].coef_
print("--- Task 4 Output ---")
print("Model Coefficients:", coefficients)

max_abs_idx = np.argmax(np.abs(coefficients))
print(f"Feature {max_abs_idx} has the largest absolute coefficient ({coefficients[max_abs_idx]:.4f}).")
# This implies that Feature 2 exerts the strongest linear influence on the target variable relative to the other features.