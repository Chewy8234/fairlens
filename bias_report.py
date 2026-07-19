import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("ai_hiring_results.csv")
df["Gender"] = df["Gender"].str.strip().str.title()

MALE_COLOR   = "#2E86C1"   # blue
FEMALE_COLOR = "#E74C3C"   # red/coral

# ── 1. Terminal report ───────────────────────────────────────────────────────
total = len(df)
male_df   = df[df["Gender"] == "Male"]
female_df = df[df["Gender"] == "Female"]

def stats(grp):
    hired     = (grp["AI_decision"] == 1).sum()
    not_hired = (grp["AI_decision"] == 0).sum()
    rate      = hired / len(grp) * 100
    return hired, not_hired, rate

m_hired, m_not, m_rate = stats(male_df)
f_hired, f_not, f_rate = stats(female_df)
gap = m_rate - f_rate

SEP = "=" * 60

if gap > 0:
    direction = "favors male applicants"
elif gap < 0:
    direction = "favors female applicants"
else:
    direction = "shows no gender gap"

if abs(gap) < 2:
    severity = "negligible"
elif abs(gap) < 5:
    severity = "modest"
elif abs(gap) < 10:
    severity = "notable"
elif abs(gap) < 15:
    severity = "significant"
else:
    severity = "severe"

report = f"""
{SEP}
   AI HIRING BIAS REPORT — GENDER ANALYSIS
{SEP}

  Total candidates tested : {total:,}

  MALE CANDIDATES ({len(male_df):,} total)
    Hired     : {m_hired:,}  ({m_rate:.1f}%)
    Not hired : {m_not:,}  ({100 - m_rate:.1f}%)

  FEMALE CANDIDATES ({len(female_df):,} total)
    Hired     : {f_hired:,}  ({f_rate:.1f}%)
    Not hired : {f_not:,}  ({100 - f_rate:.1f}%)

  GENDER BIAS GAP : {gap:+.1f} percentage points
  (Male hire rate minus Female hire rate)

{SEP}
  WHAT THIS MEANS
{SEP}

  The AI hiring model shows a {severity} gender bias that {direction}.

  Out of every 100 male applicants, roughly {m_rate:.0f} are recommended
  for hire. For every 100 female applicants, only {f_rate:.0f} are
  recommended — a gap of {abs(gap):.1f} percentage points.

  In a real hiring pipeline this means equally qualified women are
  being systematically passed over at a higher rate than men.
  This is a textbook example of algorithmic bias: the model has
  likely learned historical hiring patterns where men were hired
  more often, and is now reproducing that disparity at scale.

  Even a {abs(gap):.0f}-point gap, applied across thousands of
  applications, can meaningfully reduce diversity and expose an
  organisation to legal and reputational risk.

{SEP}
"""

print(report)

with open("results_summary.txt", "w") as f:
    f.write(report)
print("  Report saved to: results_summary.txt\n")

# ── Helpers ──────────────────────────────────────────────────────────────────
def hire_rate_by_group(data, group_col):
    """Return pivot: index=group, columns=Gender, values=hire_rate (0–100)."""
    tbl = (
        data.groupby([group_col, "Gender"])["AI_decision"]
        .agg(hired="sum", total="count")
        .reset_index()
    )
    tbl["rate"] = tbl["hired"] / tbl["total"] * 100
    pivot = tbl.pivot(index=group_col, columns="Gender", values="rate")
    return pivot

def bar_chart(pivot, title, xlabel, filename,
              figsize=(10, 6), rotate=False, bar_width=0.35):
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")

    groups   = pivot.index.tolist()
    x        = np.arange(len(groups))
    width    = bar_width

    male_vals   = pivot["Male"].values   if "Male"   in pivot.columns else np.zeros(len(groups))
    female_vals = pivot["Female"].values if "Female" in pivot.columns else np.zeros(len(groups))

    bars_m = ax.bar(x - width / 2, male_vals,   width, color=MALE_COLOR,
                    label="Male",   edgecolor="white", linewidth=0.8, zorder=3)
    bars_f = ax.bar(x + width / 2, female_vals, width, color=FEMALE_COLOR,
                    label="Female", edgecolor="white", linewidth=0.8, zorder=3)

    # value labels
    for bar in bars_m:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=MALE_COLOR)
    for bar in bars_f:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=FEMALE_COLOR)

    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel("Hire Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30 if rotate else 0,
                       ha="right" if rotate else "center", fontsize=11)
    if len(groups) == 1:
        ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, min(100, max(male_vals.max(), female_vals.max()) + 12))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    male_patch   = mpatches.Patch(color=MALE_COLOR,   label="Male")
    female_patch = mpatches.Patch(color=FEMALE_COLOR, label="Female")
    ax.legend(handles=[male_patch, female_patch], fontsize=11,
              loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")

# ── Chart 1 — Overall gender hire rates ─────────────────────────────────────
overall = pd.DataFrame({
    "Male":   [m_rate],
    "Female": [f_rate],
}, index=["All Candidates"])

bar_chart(
    overall,
    title="AI Hiring Model — Hire Rate by Gender",
    xlabel="",
    filename="bias_gender_chart.png",
    figsize=(7, 6),
    bar_width=0.2,
)

# ── Chart 2 — Hire rate by job field ─────────────────────────────────────────
field_pivot = hire_rate_by_group(df, "Job")
field_pivot = field_pivot.reindex(sorted(field_pivot.index))

bar_chart(
    field_pivot,
    title="Gender Bias by Job Field\n(Hire Rate Comparison — Male vs Female)",
    xlabel="Job Field",
    filename="bias_by_field.png",
    figsize=(12, 7),
    rotate=True,
)

# ── Chart 3 — Hire rate by university tier ───────────────────────────────────
tier_map = {1: "Low  (Vocational/\nAssociate's)",
            3: "Mid  (Bachelor's)",
            4: "High (Master's+)"}
df["Uni_Tier"] = df["Edu_Rank"].map(tier_map)

tier_pivot = hire_rate_by_group(df, "Uni_Tier")
# Sort by education level
ordered = [tier_map[k] for k in [1, 3, 4] if tier_map[k] in tier_pivot.index]
tier_pivot = tier_pivot.reindex(ordered)

bar_chart(
    tier_pivot,
    title="Gender Bias by University / Education Tier\n(Hire Rate Comparison — Male vs Female)",
    xlabel="Education Tier",
    filename="bias_by_university_tier.png",
    figsize=(10, 7),
)

print("\nAll charts saved successfully.\n")
