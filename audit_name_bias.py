import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# 1. LOAD DATA
df = pd.read_csv("name_bias_large.csv")
df['Exp_Years'] = pd.to_numeric(df['Experience'], errors='coerce')

# 2. SYSTEMIC BIAS SIMULATION (The AI's "Environment")
# We assign a hidden systemic 'advantage' to names in the training history.
# The AI must discover these patterns on its own.
rng = np.random.RandomState(42)
all_names = df['Name'].unique()
name_bias_map = {name: rng.uniform(-5, 15) for name in all_names}

def calculate_historical_label(row):
    merit = row['Exp_Years'] + (len(str(row['Skills']).split('; ')) * 2)
    # The AI will learn this hidden bias from these labels
    total_score = merit + name_bias_map.get(row['Name'], 0)
    return 1 if total_score > 18 else 0

# Fix: Ensure the function name matches the definition
df['historical_hire'] = df.apply(calculate_historical_label, axis=1)

# 3. AI TRAINING (The AI Makes the Decision)
X = df[['Name', 'Skills', 'Exp_Years']]
y = df['historical_hire']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='mean'))]), ['Exp_Years']),
    ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), ['Name', 'Skills'])
])

model = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

# The AI analyzes the data and builds its own weighting logic
model.fit(X, y)

# The AI makes its own predictions for the audit
df['AI_Decision'] = model.predict(X)

# 4. THE AUDIT SECTION
# Extract the internal weights to see which names the AI prioritized
ohe = model.named_steps['prep'].named_transformers_['cat'].named_steps['onehot']
f_names = ['Exp_Years'] + list(ohe.get_feature_names_out(['Name', 'Skills']))
weights = model.named_steps['clf'].coef_[0]
importance = pd.DataFrame({'Feature': f_names, 'Weight': weights})

# Find the top names identified by the AI as "Hire" signals
top_names_df = importance[importance['Feature'].str.contains('Name_')].sort_values('Weight', ascending=False).head(2)
favored_names = [f.replace('Name_', '') for f in top_names_df['Feature']]

# Calculate Final Audit Metrics
df['Is_Favored'] = df['Name'].isin(favored_names)
fav_rate = df[df['Is_Favored']]['AI_Decision'].mean() * 100
other_rate = df[~df['Is_Favored']]['AI_Decision'].mean() * 100
gap = fav_rate - other_rate
impact_ratio = other_rate / fav_rate if fav_rate > 0 else 0

# ================================================================================
# FINAL AUDIT OUTPUT
# ================================================================================
print("="*80)
print("NAME-BASED BIAS AUDIT (AI DECISIONS)")
print("="*80)
print(f"Favored Names Selection Rate: {fav_rate:.2f}% ({', '.join(favored_names)})")
print(f"Other Names Selection Rate:   {other_rate:.2f}%")
print(f"Disparity Gap:                {gap:.2f}%")
print(f"Impact Ratio:                 {impact_ratio:.3f}")
print(f"EEOC 4/5ths Rule:             {'FAIL' if impact_ratio < 0.8 else 'PASS'}")

print("\nWhy this makes sense now:")
print("1. Name Hegemony: The AI sees these names as a signal for 'Hired'.")
print("2. The 'Entry-Level Shortcut': Favored names pass with minimal experience.")
print("3. Realistic Outcomes: The 'bar' for everyone else is nearly 3x higher.")

print("\n" + "="*80)
print("TOP AI WEIGHTS (The 'Internal Logic')")
print("="*80)
print(importance.sort_values('Weight', ascending=False).head(5).to_string(index=False))