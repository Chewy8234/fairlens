import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Try-except for environments where fairlens might be missing
try:
    import fairlens as fl
except ImportError:
    fl = None

warnings.filterwarnings("ignore")

# ETHNICITY DEFINITIONS MAPPING
ETHNICITY_MAP = {
    'AF': 'African / African-Descent',
    'WE': 'White European',
    'BA': 'Black / African-American',
    'EA': 'East Asian',
    'H': 'Hispanic / Latino',
    'WA': 'West Asian'
}

# 1. LOAD AND CLEAN ALL DATA
df = pd.read_csv("gender_biased_final.csv")
df.columns = [c.strip() for c in df.columns]

# Create Binary Target: 1 = Good/Very Good, 0 = Bad/Very Bad
df['shortlisted'] = df['Fit'].str.lower().str.strip().isin(['good', 'very good']).astype(int)
df['Gender'] = df['Gender'].str.strip().str.capitalize()

# Add Full Ethnicity Name for clarity
df['Ethnicity_Full'] = df['Ethnicity'].str.strip().map(ETHNICITY_MAP).fillna(df['Ethnicity'])

# Tag [AF] in names for visibility as requested
df['Name'] = df.apply(lambda row: f"{str(row['Name']).strip()} [AF]" if str(row['Ethnicity']).strip() == 'AF' else str(row['Name']).strip(), axis=1)

# Extract Merit Factors (GPA, Exp, Edu, Awards, Skills)
df['GPA_numeric'] = pd.to_numeric(df['GPA'], errors='coerce')

def extract_years(text):
    match = re.search(r'(\d+)\s*year', str(text), re.IGNORECASE)
    return int(match.group(1)) if match else 0
df['Exp_Years'] = df['Work Experience'].apply(extract_years)

def rank_edu(text):
    text = str(text).lower()
    if 'phd' in text or 'doctor' in text: return 5
    if 'master' in text: return 4
    if 'bachelor' in text: return 3
    return 1
df['Edu_Rank'] = df['Education'].apply(rank_edu)

# 2. SEGMENTATION
phd_df = df[df['Edu_Rank'] == 5].reset_index(drop=True)
senior_df = df[df['Exp_Years'] >= 5].reset_index(drop=True)
high_gpa_df = df[df['GPA_numeric'] >= 3.5].reset_index(drop=True)

# 3. CALCULATE SELECTION RATES
def get_rate(data, gender):
    subset = data[data['Gender'] == gender]
    if len(subset) == 0: return 0.0
    return subset['shortlisted'].mean() * 100

overall_male = get_rate(df, 'Male')
overall_female = get_rate(df, 'Female')
phd_male = get_rate(phd_df, 'Male')
phd_female = get_rate(phd_df, 'Female')
high_gpa_male = get_rate(high_gpa_df, 'Male')
high_gpa_female = get_rate(high_gpa_df, 'Female')

# 4. PRINT ENHANCED REPORT
print("="*85)
print("AUDIT REPORT: SYSTEMIC BIAS WITH FULL ETHNICITY DEFINITIONS")
print("="*85)
print("ETHNICITY KEY:")
for code, full_name in ETHNICITY_MAP.items():
    print(f"  {code:<3} -> {full_name}")
print("-" * 85)

print(f"{'CATEGORY':<25} | {'MALE RATE':<12} | {'FEMALE RATE':<12} | {'FAVORED GROUP'}")
print("-" * 80)
print(f"{'Overall (All Data)':<25} | {overall_male:>11.2f}% | {overall_female:>11.2f}% | {'MALE' if overall_male > overall_female else 'FEMALE'}")
print(f"{'High GPA (>= 3.5)':<25} | {high_gpa_male:>11.2f}% | {high_gpa_female:>11.2f}% | {'MALE' if high_gpa_male > high_gpa_female else 'FEMALE'}")
print(f"{'PhD Holders Only':<25} | {phd_male:>11.2f}% | {phd_female:>11.2f}% | {'MALE' if phd_male > phd_female else 'FEMALE'}")

print("\n--- Intersectional Bias (Ethnicity Full Name + Gender) ---")
# Use the full name in the pivot table for clarity
intersectional = df.groupby(['Ethnicity_Full', 'Gender'])['shortlisted'].mean().unstack() * 100
print(intersectional.round(2))

# 5. VISUALIZATION
plt.figure(figsize=(12, 7))
plot_data = pd.DataFrame({
    'Category': ['Overall', 'Overall', 'High GPA', 'High GPA', 'PhD', 'PhD'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Rate': [overall_male, overall_female, high_gpa_male, high_gpa_female, phd_male, phd_female]
})
sns.barplot(x='Category', y='Rate', hue='Gender', data=plot_data, palette='viridis')
plt.title("Hiring Disparity: Men Favored Across All Merit Levels", fontsize=14)
plt.ylabel("Selection Rate (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("systemic_bias_defined_audit.png")

# Save final modified CSV
df.to_csv("final_audit_with_definitions.csv", index=False)

print("\n" + "="*85)
print("CONCLUSION: Audit complete. 'final_audit_with_definitions.csv' contains full ethnicity names.")
print("The visuals and report now explicitly define 'AF', 'WE', etc.")
print("="*85)