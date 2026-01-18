# ==========================================
# Workshop 4
# ==========================================

import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Start timer
# -----------------------------
start_time = time.time()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df.iloc[:, :2]
df.columns = ["label", "text"]

print("\nDataset preview:")
print(df.head())

# -----------------------------
# Encode labels
# ham = 0, spam = 1
# -----------------------------
le = LabelEncoder()
y = le.fit_transform(df["label"])

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_tfidf = tfidf.fit_transform(df["text"])
print("\nTF-IDF matrix shape:", X_tfidf.shape)

# -----------------------------
# Variance Threshold Feature Selection
# -----------------------------
removed_features = 0

for threshold in [0.1, 0.01, 0.0001]:
    try:
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X_tfidf)
        removed_features = X_tfidf.shape[1] - X_selected.shape[1]
        print(f"\nVariance Threshold = {threshold}")
        print("Removed features:", removed_features)
        break
    except ValueError:
        print(f"\nNo feature meets variance threshold = {threshold}")

print("Final feature matrix shape:", X_selected.shape)

# -----------------------------
# Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.3,
    shuffle=False,
    random_state=1234
)

print("\nTrain matrix shape:", X_train.shape)
print("Test matrix shape:", X_test.shape)

# -----------------------------
# Top 10 / Bottom 10 rows
# -----------------------------
print("\nTop 10 rows:")
print(df.head(10))

print("\nBottom 10 rows:")
print(df.tail(10))

# -----------------------------
# Classification (KNN)
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# Runtime
# -----------------------------
print("\nRuntime (seconds):", round(time.time() - start_time, 2))
