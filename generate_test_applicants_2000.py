"""
Generates test_applicants_2000.csv by resampling directly from
gender_biased_final.csv (1000 male, 1000 female, with replacement).
No hardcoded templates or name lists — all content comes from the
source data. Fit column is cleared so the model makes the decision.
"""
import pandas as pd
import numpy as np

np.random.seed(7)

SOURCE = "ai-hiring-model/gender_biased_final.csv"

df = pd.read_csv(SOURCE)
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

male   = df[df["Gender"] == "male"].sample(n=1000, replace=True, random_state=7)
female = df[df["Gender"] == "female"].sample(n=1000, replace=True, random_state=7)

out = pd.concat([male, female]).sample(frac=1, random_state=7).reset_index(drop=True)

out["Fit"] = ""

out.to_csv("test_applicants_2000.csv", index=False)
print(f"Saved test_applicants_2000.csv  ({len(out)} rows)")
print(out["Gender"].value_counts().to_string())
print(out["Job"].value_counts().to_string())
