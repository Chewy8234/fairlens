import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("male_dominated_large.csv")
df.columns = [c.strip() for c in df.columns]
df['Gender'] = df['Gender'].astype(str).str.strip().str.capitalize()
df['Exp_Years'] = pd.to_numeric(df['Experience'], errors='coerce')

# ============================================================
# 2. CREATE THE BIASED HISTORY (Training Data)
# ============================================================
all_tech_skills = ['Networking', 'Python', 'DevOps', 'Data Analysis', 'SQL', 'React', 'Cybersecurity', 'Machine Learning', 'Cloud', 'Java']

def historical_logic(row):
    score = row['Exp_Years'] # 1 pt per year
    candidate_skills = str(row['Skills']).split('; ')
    score += sum(2 for s in candidate_skills if s.strip() in all_tech_skills) # 2 pts per skill
    if row['Gender'] == 'Male': score += 7 # The "Invisible" Gender Bonus
    return 1 if score >= 14 else 0

df['label'] = df.apply(historical_logic, axis=1)

# ============================================================
# 3. THE AI TRAINS & MAKES DECISIONS
# ============================================================
X = df[['Skills', 'Exp_Years', 'Gender']]
y = df['label']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='mean'))]), ['Exp_Years']),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), ['Skills', 'Gender'])
])

model = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

# AI trains on the historical patterns
model.fit(X, y)

# AI now makes its own decisions for the audit
df['AI_Decision'] = model.predict(X)

# ============================================================
# 4. AUDIT & FEATURE WEIGHTS
# ============================================================
m_rate = df[df['Gender'] == 'Male']['AI_Decision'].mean() * 100
f_rate = df[df['Gender'] == 'Female']['AI_Decision'].mean() * 100
impact_ratio = f_rate / m_rate if m_rate > 0 else 0

# Extracting the AI's internal weights
ohe = model.named_steps['prep'].named_transformers_['cat'].named_steps['onehot']
feat_names = ['Exp_Years'] + list(ohe.get_feature_names_out(['Skills', 'Gender']))
weights = model.named_steps['clf'].coef_[0]
importance = pd.DataFrame({'Feature': feat_names, 'Weight': weights}).sort_values('Weight', ascending=False)

# ============================================================
# OUTPUT
# ============================================================
print("\n" + "="*80)
print("MALE-DOMINATED DATASET AUDIT (AI DECISIONS)")
print("="*80)
print(f"Male Selection Rate:   {m_rate:.2f}%")
print(f"Female Selection Rate: {f_rate:.2f}%")
print(f"Disparity Gap:         {m_rate - f_rate:.2f}%")
print(f"Impact Ratio:          {impact_ratio:.3f}")
print(f"EEOC 4/5ths Rule:      {'FAIL' if impact_ratio < 0.8 else 'PASS'}")

print("\nWhy this makes sense now:")
print(f"1. Skill Diversity: The AI scans for {len(all_tech_skills)} different tech skills.")
print("2. The 'Experience Tax': A woman must have ~7 more years of experience")
print("   than a man to get the same 'Hired' decision.")
print("3. Realistic Outcomes: Females aren't at 0%, but they are significantly")
print("   under-selected despite having the same average skills as men.")

print("\n" + "="*80)
print("TOP AI WEIGHTS (The 'Internal Logic')")
print("="*80)
print(importance.head(5).to_string(index=False))

print("\nBOTTOM AI WEIGHTS (What the AI penalizes)")
print(importance.tail(5).to_string(index=False))
print("="*80)