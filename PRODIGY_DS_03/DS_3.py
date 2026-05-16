# ==========================================================
# TASK 03: DECISION TREE CLASSIFIER - BANK MARKETING
# SkillCraft Technology Internship
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================================
# STEP 1: LOAD DATASET
# ==========================================================
df = pd.read_csv('bank-full.csv', sep=';')
print("✅ Dataset Loaded Successfully!")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

# ==========================================================
# STEP 2: EXPLORE DATA
# ==========================================================
print("\n--- 📋 First 5 Rows ---")
print(df.head())

print("\n--- 📋 Column Info ---")
print(df.dtypes)

print("\n--- 📋 Target Variable (y) ---")
print(df['y'].value_counts())

# Target Distribution Plot
plt.figure(figsize=(6, 4))
sns.countplot(x='y', data=df, palette='Set2')
plt.title('Target Distribution: Did Customer Subscribe?')
plt.xlabel('Subscribed (yes/no)')
plt.ylabel('Count')
plt.show()

# Age Distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='age', hue='y', kde=True, bins=30, palette='Set1')
plt.title('Age Distribution by Subscription')
plt.show()

# ==========================================================
# STEP 3: DATA PREPROCESSING
# ==========================================================
# Convert text columns to numbers
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

print("\n✅ All text columns converted to numbers!")
print(df.head())

# ==========================================================
# STEP 4: SPLIT DATA
# ==========================================================
X = df.drop('y', axis=1)   # Features (input)
y = df['y']                 # Target (output)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Data Split Done!")
print(f"Training Data: {X_train.shape[0]} rows")
print(f"Testing Data: {X_test.shape[0]} rows")

# ==========================================================
# STEP 5: BUILD DECISION TREE MODEL
# ==========================================================
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

model.fit(X_train, y_train)
print("\n✅ Decision Tree Model Trained Successfully!")

# ==========================================================
# STEP 6: PREDICTIONS
# ==========================================================
y_pred = model.predict(X_test)

# ==========================================================
# STEP 7: MODEL EVALUATION
# ==========================================================
accuracy = accuracy_score(y_test, y_pred) * 100

print("\n" + "="*50)
print("📊 MODEL PERFORMANCE RESULTS")
print("="*50)
print(f"\n🎯 Accuracy: {accuracy:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# ==========================================================
# STEP 8: CONFUSION MATRIX
# ==========================================================
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==========================================================
# STEP 9: DECISION TREE VISUALIZATION
# ==========================================================
plt.figure(figsize=(25, 12))
plot_tree(model,
          feature_names=X.columns,
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True,
          fontsize=8,
          max_depth=3)
plt.title('Decision Tree Visualization', fontsize=20)
plt.tight_layout()
plt.show()

# ==========================================================
# STEP 10: FEATURE IMPORTANCE
# ==========================================================
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis')
plt.title('Feature Importance (Which features matter most?)')
plt.tight_layout()
plt.show()

print("\n🏆 Top 5 Most Important Features:")
print(feature_imp.head())

print("\n" + "="*50)
print("✅ TASK 03 COMPLETED SUCCESSFULLY! 🎉")
print("="*50)
