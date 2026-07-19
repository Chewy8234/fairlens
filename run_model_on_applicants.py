"""
Loads test_applicants.csv, runs hiring_model.pkl, and reports results
with a full gender-bias breakdown.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder

# ── Load test applicants ──────────────────────────────────────────────────────
df = pd.read_csv("test_applicants.csv")
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
print(f"Loaded {len(df)} applicants from test_applicants.csv\n")

# ── Load model bundle ─────────────────────────────────────────────────────────
with open("hiring_model.pkl", "rb") as f:
    bundle = pickle.load(f)
clf   = bundle["model"]
tfidf = bundle["tfidf"]

# ── Fit encoders on training data (same as training) ─────────────────────────
train = pd.read_csv("ai-hiring-model/gender_biased_final.csv")
train = train.map(lambda x: x.strip() if isinstance(x, str) else x)

gender_le   = LabelEncoder().fit(train["Gender"])
ethnicity_le = LabelEncoder().fit(train["Ethnicity"])
job_le       = LabelEncoder().fit(train["Job"])

# ── Feature engineering ───────────────────────────────────────────────────────
df["GPA_num"]      = pd.to_numeric(df["GPA"], errors="coerce").fillna(0)
df["Gender_enc"]   = gender_le.transform(df["Gender"])
df["Ethnicity_enc"] = ethnicity_le.transform(df["Ethnicity"])
df["Job_enc"]       = job_le.transform(df["Job"])
df["text_combined"] = (df["Education"].fillna("") + " " +
                       df["Work Experience"].fillna("") + " " +
                       df["Skills"].fillna("") + " " +
                       df["Awards"].fillna(""))

num_cols = ["Gender_enc", "Ethnicity_enc", "Job_enc", "GPA_num"]
X_num   = csr_matrix(df[num_cols].values)
X_text  = tfidf.transform(df["text_combined"])
X_final = hstack([X_num, X_text])

# ── Predict ───────────────────────────────────────────────────────────────────
df["Predicted_Fit"] = clf.predict(X_final)
df["Hired"]         = df["Predicted_Fit"].isin(["good", "very good"])

# Save enriched predictions
df[["Name","Gender","Ethnicity","Job","GPA","Predicted_Fit","Hired"]].to_csv(
    "applicant_predictions.csv", index=False
)

# ── Console report ────────────────────────────────────────────────────────────
W = 58
print("=" * W)
print("   HIRING MODEL RESULTS — test_applicants.csv (n=200)")
print("=" * W)

total    = len(df)
hired    = df["Hired"].sum()
not_hired = total - hired

print(f"\n  Total applicants : {total}")
print(f"  Hired            : {hired}  ({hired/total*100:.1f}%)")
print(f"  Not hired        : {not_hired}  ({not_hired/total*100:.1f}%)")

# ── Gender breakdown ──────────────────────────────────────────────────────────
print(f"\n{'─'*W}")
print("  GENDER BREAKDOWN")
print(f"{'─'*W}")

g = df.groupby("Gender").agg(
    Applicants=("Hired","count"),
    Hired=("Hired","sum"),
    Not_Hired=("Hired", lambda x: (~x).sum()),
).assign(Hire_Rate=lambda d: (d["Hired"]/d["Applicants"]*100).round(1))

print(g.to_string())

male_rate   = g.loc["male",   "Hire_Rate"]
female_rate = g.loc["female", "Hire_Rate"]
gap = male_rate - female_rate

print(f"\n  Male hire rate   : {male_rate:.1f}%")
print(f"  Female hire rate : {female_rate:.1f}%")
print(f"  Gap (M − F)      : {gap:+.1f} percentage points")

print(f"\n{'─'*W}")
if abs(gap) >= 10:
    print(f"  *** SIGNIFICANT GENDER BIAS DETECTED ({abs(gap):.1f} pp gap) ***")
elif abs(gap) >= 5:
    print(f"  ** Moderate gender bias detected ({abs(gap):.1f} pp gap).")
else:
    print(f"  No significant gender bias (<5 pp gap).")
print(f"{'─'*W}")

# ── Predicted Fit distribution by gender ─────────────────────────────────────
print(f"\n  PREDICTED FIT DISTRIBUTION BY GENDER")
fit_order = ["very bad","bad","average","good","very good"]
fit_counts = df.groupby(["Gender","Predicted_Fit"]).size().unstack(fill_value=0)
for c in fit_order:
    if c not in fit_counts.columns:
        fit_counts[c] = 0
print(fit_counts[fit_order].to_string())

# ── Per-job breakdown ─────────────────────────────────────────────────────────
print(f"\n  HIRE RATE BY JOB AND GENDER")
job_g = df.groupby(["Job","Gender"]).agg(
    n=("Hired","count"),
    hired=("Hired","sum"),
).assign(rate=lambda d: (d["hired"]/d["n"]*100).round(1))
print(job_g.to_string())

print(f"\n  Full predictions saved → applicant_predictions.csv")

# ── Charts ────────────────────────────────────────────────────────────────────
PALETTE = {"male": "#4C72B0", "female": "#DD8452"}
FIT_COLORS = ["#d62728","#ff7f0e","#bcbd22","#2ca02c","#17becf"]

fig = plt.figure(figsize=(16, 11))
fig.suptitle("AI Hiring Model — Gender Bias Analysis (200 Applicants)",
             fontsize=14, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

# 1. Overall hired/not hired
ax1 = fig.add_subplot(gs[0, 0])
ax1.pie([hired, not_hired], labels=["Hired","Not Hired"],
        colors=["#2ca02c","#d62728"], autopct="%1.1f%%", startangle=90,
        textprops={"fontsize":11})
ax1.set_title("Overall: Hired vs Not Hired", fontweight="bold")

# 2. Hired count by gender
ax2 = fig.add_subplot(gs[0, 1])
g_reset = g.reset_index()
bars = ax2.bar(g_reset["Gender"], g_reset["Hired"],
               color=[PALETTE[x] for x in g_reset["Gender"]], width=0.5)
ax2.set_title("Candidates Hired by Gender", fontweight="bold")
ax2.set_ylabel("# Hired")
ax2.set_ylim(0, max(g["Hired"]) * 1.2)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             int(bar.get_height()), ha="center", fontsize=12, fontweight="bold")

# 3. Hire rate by gender
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.bar(g_reset["Gender"], g_reset["Hire_Rate"],
                color=[PALETTE[x] for x in g_reset["Gender"]], width=0.5)
ax3.set_title("Hire Rate by Gender (%)", fontweight="bold")
ax3.set_ylabel("Hire Rate (%)")
ax3.set_ylim(0, 110)
ax3.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% line")
for bar in bars3:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{bar.get_height():.1f}%", ha="center", fontsize=11, fontweight="bold")
# annotate gap arrow
y_m = g.loc["male","Hire_Rate"]
y_f = g.loc["female","Hire_Rate"]
ax3.annotate("", xy=(1, y_f), xytext=(1, y_m),
             arrowprops=dict(arrowstyle="<->", color="red", lw=2))
ax3.text(1.18, (y_m+y_f)/2, f"{gap:+.1f}pp\ngap", color="red", fontsize=9, va="center")

# 4. Predicted fit distribution (grouped bar)
ax4 = fig.add_subplot(gs[1, 0:2])
fit_pct = fit_counts[fit_order].div(fit_counts[fit_order].sum(axis=1), axis=0) * 100
x = np.arange(len(fit_order))
width = 0.35
genders = fit_pct.index.tolist()
for i, gen in enumerate(genders):
    offset = (i - 0.5) * width
    rects = ax4.bar(x + offset, fit_pct.loc[gen, fit_order], width,
                    label=gen, color=PALETTE[gen], alpha=0.9)
ax4.set_title("Predicted Fit Distribution by Gender (%)", fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels(fit_order)
ax4.set_ylabel("% of Gender Group")
ax4.legend(title="Gender")
ax4.set_ylim(0, max(fit_pct.values.flatten()) * 1.25)

# 5. Hire rate by job & gender (horizontal bar)
ax5 = fig.add_subplot(gs[1, 2])
jr = df.groupby(["Job","Gender"])["Hired"].mean().reset_index()
jr["Hired"] *= 100
sns.barplot(data=jr, x="Hired", y="Job", hue="Gender",
            palette=PALETTE, orient="h", ax=ax5)
ax5.set_title("Hire Rate by Job & Gender (%)", fontweight="bold")
ax5.set_xlabel("Hire Rate (%)")
ax5.set_ylabel("")
ax5.set_xlim(0, 115)
ax5.legend(title="Gender", loc="lower right", fontsize=8)
ax5.axvline(50, color="gray", linestyle="--", linewidth=0.8)

plt.savefig("applicant_bias_report.png", dpi=150, bbox_inches="tight")
print("  Chart saved → applicant_bias_report.png")
