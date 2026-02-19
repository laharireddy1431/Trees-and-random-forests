# Trees-and-random-forests
In this task, we implemented tree-based machine learning models for classification using Python and Scikit-learn. A Decision Tree classifier was trained and visualized to understand how the model makes decisions. Overfitting was analyzed by controlling the tree depth. A Random Forest classifier was then trained to improve performance .
# ============================================
# Task 5: Decision Trees and Random Forests
# ============================================

# 1. Import required libraries
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ============================================
# 2. Load Dataset
# Using Breast Cancer Dataset (built-in)
# ============================================

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset shape:", X.shape)

# ============================================
# 3. Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 4. Decision Tree Classifier
# ============================================

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predictions
y_pred_dt = dt.predict(X_test)

# Accuracy
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("\nDecision Tree Accuracy:", dt_accuracy)

# ============================================
# 5. Control Overfitting (Limit Tree Depth)
# ============================================

dt_limited = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_limited.fit(X_train, y_train)

y_pred_limited = dt_limited.predict(X_test)
limited_accuracy = accuracy_score(y_test, y_pred_limited)

print("Decision Tree (Max Depth = 4) Accuracy:", limited_accuracy)

# ============================================
# 6. Visualize Decision Tree
# ============================================

plt.figure(figsize=(20, 10))
plot_tree(
    dt_limited,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True
)
plt.title("Decision Tree Visualization (Max Depth = 4)")
plt.show()

# ============================================
# 7. Random Forest Classifier
# ============================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:", rf_accuracy)

# ============================================
# 8. Feature Importance
# ============================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance["Feature"][:10],
    feature_importance["Importance"][:10]
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.gca().invert_yaxis()
plt.show()

# ============================================
# 9. Cross-Validation
# ============================================

cv_scores_dt = cross_val_score(dt_limited, X, y, cv=5)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)

print("\nCross-Validation Accuracy (Decision Tree):", cv_scores_dt.mean())
print("Cross-Validation Accuracy (Random Forest):", cv_scores_rf.mean())

# ============================================
# 10. Final Comparison
# ============================================

print("\n--- Model Comparison ---")
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree (Limited Depth) Accuracy:", limited_accuracy)
print("Random Forest Accuracy:", rf_accuracy)