
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix

CSV_PATH   = "ai-hiring-model/gender_biased_final.csv"
MODEL_PATH = "hiring_model.pkl"

# --- Load & clean ---
df = pd.read_csv(CSV_PATH)
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['Fit'].value_counts()}\n")

# --- Feature engineering ---
df["GPA_num"] = pd.to_numeric(df["GPA"], errors="coerce").fillna(0)
df["text_combined"] = (
    df["Education"].fillna("")       + " " +
    df["Work Experience"].fillna("") + " " +
    df["Skills"].fillna("")          + " " +
    df["Awards"].fillna("")
)

for col in ["Gender", "Ethnicity", "Job"]:
    df[col] = df[col].fillna("unknown")

df["Gender_enc"]    = LabelEncoder().fit_transform(df["Gender"])
df["Ethnicity_enc"] = LabelEncoder().fit_transform(df["Ethnicity"])
df["Job_enc"]       = LabelEncoder().fit_transform(df["Job"])

# --- Build feature matrix ---
num_cols = ["Gender_enc", "Ethnicity_enc", "Job_enc", "GPA_num"]
X = df[num_cols + ["text_combined"]]
y = df["Fit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf         = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train["text_combined"])
X_test_tfidf  = tfidf.transform(X_test["text_combined"])

X_train_num   = csr_matrix(X_train[num_cols].values)
X_test_num    = csr_matrix(X_test[num_cols].values)

X_train_final = hstack([X_train_num, X_train_tfidf])
X_test_final  = hstack([X_test_num,  X_test_tfidf])

# --- Train ---
print("Training RandomForest ...")
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train_final, y_train)

# --- Held-out accuracy ---
y_pred = clf.predict(X_test_final)
acc    = accuracy_score(y_test, y_pred)
print(f"\nHeld-out test accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
print(classification_report(y_test, y_pred))

# --- Confusion matrix ---
labels = ["very bad", "bad", "average", "good", "very good"]
cm     = confusion_matrix(y_test, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — Test Accuracy: {acc*100:.2f}%")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Confusion matrix saved to confusion_matrix.png")

# --- Save model bundle ---
bundle = {
    "model":    clf,
    "tfidf":    tfidf,
    "num_cols": num_cols,
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(bundle, f)
print(f"Model saved to {MODEL_PATH}")
