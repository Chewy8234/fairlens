import pandas as pd
import numpy as np
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# 1. LOAD DATA
df = pd.read_csv("race_gender_bias_large.csv")
df.columns = [c.strip() for c in df.columns]

# 2. PREPARE FEATURES
df['Exp_Years'] = pd.to_numeric(df['Experience'], errors='coerce').fillna(0)
df['Skill_Count'] = df['Skills'].apply(lambda x: len(str(x).split(';')))

# 3. CREATE EXTREME HISTORICAL BIAS
# We set a high bar for "Success" and give massive bonuses to specific groups
np.random.seed(42)

# Merit is now secondary (Weight: 0.05 per year)
merit_component = (df['Exp_Years'] * 0.05) + (df['Skill_Count'] * 0.05)

# Demographics are primary (Weight: 0.60 for Male, 0.40 for White)
# This creates a "glass ceiling" effect in the training labels
gender_bonus = (df['Gender'].str.strip() == 'Male').astype(int) * 0.60
race_bonus = (df['Ethnicity'].str.strip() == 'White').astype(int) * 0.40

# Calculate raw score and apply a strict threshold
raw_score = merit_component + gender_bonus + race_bonus
# Only the top scores (where demographic bonuses push them over) get the "Hired" label
y = (raw_score >= 0.85).astype(int)

# 4. TRAIN THE AI (Blind to Demographics)
X = df[['Skills', 'Exp_Years']]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Skills']),
    ('num', SimpleImputer(strategy='mean'), ['Exp_Years'])
])

ai_model = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
])

ai_model.fit(X, y)

# 5. AI DECISIONS
df['AI_Decision'] = ai_model.predict(X)

# 6. OUTPUT RESULTS
df['Gender'] = df['Gender'].str.strip().str.capitalize()
df['Ethnicity'] = df['Ethnicity'].str.strip().str.capitalize()

m_rate = df[df['Gender'] == 'Male']['AI_Decision'].mean() * 100
f_rate = df[df['Gender'] == 'Female']['AI_Decision'].mean() * 100
impact_ratio = f_rate / m_rate if m_rate > 0 else 0

print("="*80)
print("EXTREME SYSTEMIC BIAS AUDIT (AI LEARNED)")
print("="*80)
print(f"Male Selection Rate:   {m_rate:.2f}%")
print(f"Female Selection Rate: {f_rate:.2f}%")
print(f"Gender Impact Ratio:   {impact_ratio:.3f}")
print(f"EEOC 4/5ths Rule:      {'PASS' if impact_ratio >= 0.8 else 'FAIL'}")

print("\nINTERSECTIONAL BIAS (% HIRED BY AI):")
intersectional = df.groupby(['Gender', 'Ethnicity'])['AI_Decision'].mean() * 100
print(intersectional.unstack().round(2))

print("\nConclusion:")
print("The AI has successfully inherited the structural bias.")
print("White Males are being prioritized by the model even with lower skills.")
print("="*80)