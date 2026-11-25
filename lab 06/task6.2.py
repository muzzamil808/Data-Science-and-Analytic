import numpy as np
from sklearn.datasets import make_moons, load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Task 1: KNN Prediction and Probability ---
print("--- Task 1 Output ---")
X_moons, y_moons = make_moons(n_samples=100, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_moons, y_moons, test_size=0.2, random_state=42)

knn_moons = KNeighborsClassifier(n_neighbors=5)
knn_moons.fit(X_train_m, y_train_m)

preds_m = knn_moons.predict(X_test_m[:3])
probs_m = knn_moons.predict_proba(X_test_m[:3])

print("Predicted classes for first 3 test samples:", preds_m)
print("Predicted probabilities for first 3 test samples:\n", probs_m)
print()

# --- Task 2: Decision Tree Depth Impact ---
print("--- Task 2 Output ---")
iris = load_iris()
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

dt_shallow = DecisionTreeClassifier(max_depth=1, random_state=0)
dt_shallow.fit(X_train_i, y_train_i)
acc_shallow = accuracy_score(y_test_i, dt_shallow.predict(X_test_i))

dt_deep = DecisionTreeClassifier(max_depth=10, random_state=0)
dt_deep.fit(X_train_i, y_train_i)
acc_deep = accuracy_score(y_test_i, dt_deep.predict(X_test_i))

print(f"Shallow Tree (max_depth=1) Test Accuracy: {acc_shallow:.4f}")
print(f"Deep Tree (max_depth=10) Test Accuracy: {acc_deep:.4f}")
print()

# --- Task 3: Logistic Regression Probability Output ---
print("--- Task 3 Output ---")
wine = load_wine()
X_w, y_w = wine.data, wine.target

lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=1, max_iter=5000)
lr.fit(X_w, y_w)

probs_w = lr.predict_proba(X_w[:2])
print("Probability matrix for first 2 samples:\n", probs_w)
for i, p in enumerate(probs_w):
    winning_class = np.argmax(p)
    print(f"Sample {i+1} winning class: {winning_class} (probability: {p[winning_class]:.4f})")
print()

# --- Task 4: KNN Overfitting Demonstration ---
print("--- Task 4 Output ---")
X_c, y_c = make_classification(n_features=5, n_informative=3, n_classes=2, random_state=10)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.3, random_state=10)

print(f"{'Model':<12} | {'Train Accuracy':<15} | {'Test Accuracy':<15}")
print("-" * 50)
for k in [1, 15]:
    knn_c = KNeighborsClassifier(n_neighbors=k)
    knn_c.fit(X_train_c, y_train_c)
    train_acc = accuracy_score(y_train_c, knn_c.predict(X_train_c))
    test_acc = accuracy_score(y_test_c, knn_c.predict(X_test_c))
    print(f"KNN (k={k:2d})   | {train_acc:.4f}          | {test_acc:.4f}          ")