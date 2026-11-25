# --- Task 1: Compute Confusion Matrix and Report ---
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== TASK 1 OUTPUT ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("-" * 50)


# --- Task 2: Imbalanced Data Metric Selection ---
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

X_imb, y_imb = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)
X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(X_imb, y_imb, test_size=0.2, stratify=y_imb, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_tr_i, y_tr_i)
y_pred_i = rf.predict(X_te_i)

print("=== TASK 2 OUTPUT ===")
print(f"Accuracy: {accuracy_score(y_te_i, y_pred_i):.4f}")
print(f"Recall (Class 1): {recall_score(y_te_i, y_pred_i, pos_label=1):.4f}")
print(f"F1-Score (Class 1): {f1_score(y_te_i, y_pred_i, pos_label=1):.4f}")
print("-" * 50)


# --- Task 3: Cross-Validation with Scoring ---
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

bc_data = load_breast_cancer()
X_bc, y_bc = bc_data.data, bc_data.target
lr_bc = LogisticRegression(max_iter=5000, random_state=42)

acc_scores = cross_val_score(lr_bc, X_bc, y_bc, cv=5, scoring='accuracy')
recall_scores = cross_val_score(lr_bc, X_bc, y_bc, cv=5, scoring='recall')

print("=== TASK 3 OUTPUT ===")
print("Accuracy Scores:", acc_scores)
print("Recall Scores:", recall_scores)
print("-" * 50)


# --- Task 4: Leak-Free Pipeline Evaluation ---
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

wine = load_wine()
mask = wine.target < 2
X_w, y_w = wine.data[mask], wine.target[mask]

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(random_state=42))
])

f1_scores = cross_val_score(pipeline, X_w, y_w, cv=3, scoring='f1')
print("=== TASK 4 OUTPUT ===")
print("Pipeline Definition:\n", pipeline)
print("F1 Scores:", f1_scores)