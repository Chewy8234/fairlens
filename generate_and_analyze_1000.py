"""
Generates test_applicants_1000.csv, runs hiring_model.pkl on it,
prints gender bias stats, and saves 3 charts.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Name / data pools ─────────────────────────────────────────────────────────
MALE_FIRST = [
    "James","Michael","David","Robert","William","Daniel","Matthew","Christopher",
    "Andrew","Joshua","Ryan","Justin","Brandon","Tyler","Nathan","Kevin","Brian",
    "Jason","Eric","Adam","Sean","Patrick","Timothy","Gregory","Jonathan","Steven",
    "Kyle","Aaron","Jeremy","Zachary","Marcus","Darius","Malik","Jamal","Antoine",
    "Carlos","Miguel","Diego","Eduardo","Luis","Wei","Ravi","Amir","Khalid","Liam",
    "Ethan","Noah","Oliver","George","Lars","Matteo","Hiroshi","Kenji","Ibrahim",
    "Kwame","Chukwudi","Arjun","Vikram","Rahul","Takeshi","Jun","Bao","Minh",
]
MALE_LAST = [
    "Carter","Brown","Kim","Chen","Davis","Martinez","Johnson","Lee","Wilson",
    "Thompson","Anderson","Taylor","Harris","Robinson","Clark","Lewis","Walker",
    "Hall","Young","Allen","Wright","Scott","Green","Adams","Baker","Nelson",
    "Mitchell","Perez","Roberts","Turner","Jackson","Brooks","Coleman","Powell",
    "Rivera","Flores","Ramirez","Ortiz","Torres","Zhang","Patel","Hassan","Rahman",
    "O'Brien","Murphy","Bennett","Thornton","Eriksson","Romano","Tanaka","Yamamoto",
    "Diallo","Mensah","Okafor","Sharma","Singh","Gupta","Nakamura","Watanabe",
    "Nguyen","Tran","Pham","Andersen","Novak","Volkov","Asante","Nwosu",
]
FEMALE_FIRST = [
    "Emily","Jessica","Sarah","Ashley","Amanda","Stephanie","Melissa","Rebecca",
    "Lauren","Rachel","Megan","Brittany","Kayla","Amber","Danielle","Crystal",
    "Shannon","Tiffany","Jennifer","Christina","Olivia","Emma","Ava","Isabella",
    "Sophia","Mia","Charlotte","Amelia","Harper","Evelyn","Aaliyah","Jasmine",
    "Keisha","Latisha","Shaniqua","Tamara","Monique","Maria","Sofia","Valentina",
    "Camila","Fernanda","Natalia","Mei","Xiao","Yu","Ling","Priya","Ananya",
    "Divya","Fatima","Layla","Nour","Siobhan","Aoife","Freya","Astrid","Chiara",
    "Giulia","Yuki","Sakura","Hana","Nadia","Elena","Amara","Ama","Ngozi","Leila",
]
FEMALE_LAST = [
    "Johnson","Williams","Davis","Brown","Wilson","Moore","Taylor","Anderson",
    "Thomas","Jackson","White","Harris","Martin","Thompson","Garcia","Martinez",
    "Robinson","Clark","Lewis","Hall","Young","Walker","Allen","Wright","Scott",
    "Green","Adams","Baker","Nelson","Mitchell","Washington","Brooks","Coleman",
    "Hughes","Foster","Powell","Rodriguez","Hernandez","Lopez","Diaz","Cruz",
    "Lin","Wang","Chen","Zhang","Patel","Sharma","Singh","Al-Hassan","Rahman",
    "Abdullah","O'Connor","Murphy","Walsh","Brennan","Larsson","Lindqvist",
    "Romano","Ferrari","Tanaka","Yamamoto","Popescu","Ionescu","Diallo","Mensah",
    "Okafor","Ahmadi","Hosseini","Park","Kim","Lee","Choi","Nguyen","Tran",
]

JOBS        = ["Hairdresser","Engineer","Nurse","Secretary","Surgeon","Construction Worker"]
ETHNICITIES = ["AF","BA","H","EA","WE","WA"]
CITIES      = ["New York","Chicago","Houston","Phoenix","Los Angeles",
               "Seattle","Boston","Atlanta","Denver","Miami"]
SALONS      = ["Trendy Cuts Salon","Elite Style Studio","The Hair Lounge",
               "Urban Chic Salon","Prestige Hair Co"]
HOSPITALS   = ["City General Hospital","Metro Medical Center","St. Luke's Hospital",
               "University Hospital","Riverside Clinic"]
COMPANIES   = ["TechCorp","BuildRight Inc","Apex Solutions","Metro Group",
               "Nexus Industries","Summit Engineering"]

EDUCATION = {
    "Hairdresser":         ["Associate Degree in Cosmetology from {city} Beauty College",
                            "Bachelor's in Cosmetology from {city} School of Cosmetology",
                            "Certificate in Hairdressing from {city} Institute of Beauty",
                            "Diploma in Barbering and Styling from {city} Vocational School"],
    "Engineer":            ["Bachelor's in Mechanical Engineering from {city} State University",
                            "Master's in Electrical Engineering from {city} Tech Institute",
                            "Bachelor's in Civil Engineering from University of {city}",
                            "PhD in Computer Engineering from {city} Institute of Technology",
                            "Bachelor's in Software Engineering from {city} Polytechnic"],
    "Nurse":               ["Bachelor's in Nursing from {city} University Medical School",
                            "Associate Degree in Nursing from {city} Community College",
                            "Master's in Nursing Practice from {city} Health Sciences University",
                            "Bachelor of Science in Nursing from {city} General Hospital School of Nursing"],
    "Secretary":           ["Associate Degree in Business Administration from {city} Community College",
                            "Bachelor's in Office Management from {city} State College",
                            "Certificate in Administrative Support from {city} Business Institute",
                            "Bachelor's in Communications from {city} University"],
    "Surgeon":             ["MD from {city} School of Medicine",
                            "PhD in Surgical Sciences from {city} Medical University",
                            "Master's in Surgery from {city} General Hospital Medical School",
                            "MD and residency training at {city} Academic Medical Center"],
    "Construction Worker": ["Certificate in Construction Technology from {city} Vocational Institute",
                            "Associate Degree in Building Trades from {city} Technical College",
                            "Bachelor's in Construction Management from {city} State University",
                            "Apprenticeship Certificate in Carpentry from {city} Trades Council"],
}

WORK_EXP = {
    "Hairdresser":         ["{name} has {n} years of experience at {salon}, specializing in precision cuts and advanced color techniques.",
                            "{name} worked for {n} years as a lead stylist at {salon}, recognized for client satisfaction and creativity.",
                            "{name} has {n} years of barbering and styling experience at {salon}, excelling in modern and classic techniques."],
    "Engineer":            ["{name} has {n} years of engineering experience at {company}, leading cross-functional technical projects.",
                            "{name} spent {n} years at {company} designing mechanical systems and managing product development cycles.",
                            "{name} has {n} years at {company} developing scalable software solutions and mentoring junior engineers."],
    "Nurse":               ["{name} has {n} years of clinical nursing experience at {hospital}, providing compassionate patient-centered care.",
                            "{name} worked for {n} years in the ICU at {hospital}, specializing in critical care and emergency response.",
                            "{name} has {n} years of experience in pediatric nursing at {hospital}, known for excellent family communication."],
    "Secretary":           ["{name} has {n} years of administrative experience supporting C-suite executives at {company}.",
                            "{name} managed all office operations for {n} years at {company}, coordinating calendars, travel, and communications.",
                            "{name} has {n} years of experience as an executive assistant at {company}, handling confidential correspondence."],
    "Surgeon":             ["{name} has {n} years of surgical experience at {hospital}, performing complex minimally invasive procedures.",
                            "{name} completed {n} years of residency and surgical practice at {hospital}, specializing in orthopedic surgery.",
                            "{name} has {n} years of experience as a cardiac surgeon at {hospital}, with an outstanding patient outcome record."],
    "Construction Worker": ["{name} has {n} years of hands-on construction experience at {company}, skilled in structural and finishing work.",
                            "{name} worked on major infrastructure projects for {n} years with {company}, operating heavy equipment safely.",
                            "{name} has {n} years of experience in residential and commercial construction at {company}."],
}

SKILLS = {
    "Hairdresser":         ["Proficient in cutting, coloring, highlighting, and styling. Strong customer service and interpersonal skills.",
                            "Skilled in balayage, keratin treatments, and precision cuts. Excellent time management in fast-paced salon environments."],
    "Engineer":            ["Skilled in CAD, project management, and systems design. Proficient in Python, MATLAB, and AutoCAD.",
                            "Expert in machine learning, data pipelines, and cloud infrastructure. Strong analytical and problem-solving skills."],
    "Nurse":               ["Expert in patient assessment, IV therapy, and medication management. Strong empathy and crisis communication.",
                            "Skilled in wound care, patient education, and electronic health records. Calm under pressure with excellent bedside manner."],
    "Secretary":           ["Proficient in MS Office Suite, scheduling software, and correspondence drafting. Excellent multitasking and discretion.",
                            "Skilled in calendar management, travel coordination, and meeting facilitation. Fast typing with superior attention to detail."],
    "Surgeon":             ["Expert in laparoscopic and robotic surgical techniques. Strong decision-making under pressure and exceptional dexterity.",
                            "Skilled in trauma surgery and post-operative care. Proficient in surgical simulation and resident teaching."],
    "Construction Worker": ["Skilled in carpentry, concrete work, scaffolding, and equipment operation. Strong safety awareness and physical endurance.",
                            "Proficient in blueprint reading, site coordination, and OSHA compliance. Reliable team player with strong work ethic."],
}

AWARDS = {
    "Hairdresser":         ["{name} received the 'Best Stylist' award at the {city} Hair and Beauty Expo.",
                            "{name} was named Top Hairdresser of the Year by {salon}."],
    "Engineer":            ["{name} received the 'Innovation Award' from the {city} Engineering Society.",
                            "{name} was recognized as Engineer of the Quarter at {company} for outstanding project delivery."],
    "Nurse":               ["{name} received the 'Excellence in Patient Care' award from {hospital}.",
                            "{name} was named Nurse of the Year at {hospital} for exceptional dedication."],
    "Secretary":           ["{name} was awarded 'Administrative Professional of the Year' at {company}.",
                            "{name} received the 'Outstanding Support Staff' recognition at the {city} Business Excellence Gala."],
    "Surgeon":             ["{name} received the 'Outstanding Surgeon' award from the {city} Medical Association.",
                            "{name} was recognized for surgical excellence and innovation at {hospital}'s annual awards ceremony."],
    "Construction Worker": ["{name} received the 'Safety Excellence' award from {company} for zero-incident project leadership.",
                            "{name} was recognized as Top Performer on the {city} Downtown Infrastructure Project."],
}

MALE_COLOR   = "#2E86C1"
FEMALE_COLOR = "#E74C3C"
SEP = "=" * 60


def pick(rng, lst):
    return lst[rng.integers(len(lst))]


def make_person(rng, gender):
    job   = pick(rng, JOBS)
    city  = pick(rng, CITIES)
    eth   = pick(rng, ETHNICITIES)
    gpa   = round(float(rng.uniform(2.4, 4.0)), 1)
    yrs   = int(rng.integers(1, 16))
    first = pick(rng, MALE_FIRST  if gender == "male" else FEMALE_FIRST)
    last  = pick(rng, MALE_LAST   if gender == "male" else FEMALE_LAST)
    name  = f"{first} {last}"
    email = f"{first.lower()}.{last.lower().replace(chr(39), '')}@email.com"
    edu   = pick(rng, EDUCATION[job]).format(city=city)
    exp   = pick(rng, WORK_EXP[job]).format(
                name=first, n=yrs, salon=pick(rng, SALONS),
                hospital=pick(rng, HOSPITALS), company=pick(rng, COMPANIES))
    skill = pick(rng, SKILLS[job])
    award = pick(rng, AWARDS[job]).format(
                name=first, city=city, salon=pick(rng, SALONS),
                hospital=pick(rng, HOSPITALS), company=pick(rng, COMPANIES))
    return {"Name": name, "Gender": gender, "Email": email, "Ethnicity": eth,
            "Education": edu, "GPA": str(gpa), "Work Experience": exp,
            "Skills": skill, "Awards": award, "Fit": "", "Job": job}


def edu_tier(text):
    if not isinstance(text, str):
        return "Unknown"
    t = text.lower()
    if any(k in t for k in ["master","mba","phd","doctorate","md ","doctor of"]):
        return "High  (Master's+)"
    if any(k in t for k in ["bachelor","b.s.","b.a.","undergraduate"]):
        return "Mid  (Bachelor's)"
    return "Low  (Vocational/\nAssociate's)"


def hire_rate_pivot(df, group_col):
    tbl = (df.groupby([group_col, "Gender"])["AI_decision"]
             .agg(hired="sum", total="count").reset_index())
    tbl["rate"] = tbl["hired"] / tbl["total"] * 100
    return tbl.pivot(index=group_col, columns="Gender", values="rate")


def bar_chart(pivot, title, xlabel, filename, figsize=(10, 6), rotate=False, bar_width=0.35):
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")
    groups      = pivot.index.tolist()
    x           = np.arange(len(groups))
    width       = bar_width
    male_vals   = pivot["Male"].values   if "Male"   in pivot.columns else np.zeros(len(groups))
    female_vals = pivot["Female"].values if "Female" in pivot.columns else np.zeros(len(groups))

    ax.bar(x - width/2, male_vals,   width, color=MALE_COLOR,   edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + width/2, female_vals, width, color=FEMALE_COLOR, edgecolor="white", linewidth=0.8, zorder=3)

    for val, xpos in zip(male_vals,   x - width/2):
        ax.text(xpos + width/2, val + 0.8, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=MALE_COLOR)
    for val, xpos in zip(female_vals, x + width/2):
        ax.text(xpos + width/2, val + 0.8, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=FEMALE_COLOR)

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
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(handles=[mpatches.Patch(color=MALE_COLOR,   label="Male"),
                        mpatches.Patch(color=FEMALE_COLOR, label="Female")],
              fontsize=11, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ── Generate CSV ──────────────────────────────────────────────────────────────
rng  = np.random.default_rng(99)
rows = [make_person(rng, "male") for _ in range(500)] + \
       [make_person(rng, "female") for _ in range(500)]
df   = pd.DataFrame(rows).sample(frac=1, random_state=99).reset_index(drop=True)
df.to_csv("test_applicants_1000.csv", index=False)
print(f"\n  Generated test_applicants_1000.csv  ({len(df)} rows)")

# ── Load model & encoders ─────────────────────────────────────────────────────
with open("hiring_model.pkl", "rb") as f:
    bundle = pickle.load(f)
clf   = bundle["model"]
tfidf = bundle["tfidf"]

train = pd.read_csv("ai-hiring-model/gender_biased_final.csv")
train = train.map(lambda x: x.strip() if isinstance(x, str) else x)
gender_le    = LabelEncoder().fit(train["Gender"])
ethnicity_le = LabelEncoder().fit(train["Ethnicity"])
job_le       = LabelEncoder().fit(train["Job"])

# ── Run model ─────────────────────────────────────────────────────────────────
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
df["GPA_num"]       = pd.to_numeric(df["GPA"], errors="coerce").fillna(0)
df["Gender_enc"]    = gender_le.transform(df["Gender"])
df["Ethnicity_enc"] = ethnicity_le.transform(df["Ethnicity"])
df["Job_enc"]       = job_le.transform(df["Job"])
df["text_combined"] = (df["Education"].fillna("") + " " + df["Work Experience"].fillna("") + " " +
                       df["Skills"].fillna("") + " " + df["Awards"].fillna(""))
X_num   = csr_matrix(df[["Gender_enc","Ethnicity_enc","Job_enc","GPA_num"]].values)
X_text  = tfidf.transform(df["text_combined"])
df["Predicted_Fit"] = clf.predict(hstack([X_num, X_text]))
df["AI_decision"]   = df["Predicted_Fit"].isin(["good","very good"]).astype(int)
df["Gender"]        = df["Gender"].str.strip().str.title()
df["Uni_Tier"]      = df["Education"].apply(edu_tier)

# ── Stats ─────────────────────────────────────────────────────────────────────
total     = len(df)
male_df   = df[df["Gender"] == "Male"]
female_df = df[df["Gender"] == "Female"]
m_hired   = (male_df["AI_decision"]   == 1).sum()
f_hired   = (female_df["AI_decision"] == 1).sum()
m_rate    = m_hired / len(male_df)   * 100
f_rate    = f_hired / len(female_df) * 100
gap       = m_rate - f_rate

if   gap > 0: direction = "favors male applicants"
elif gap < 0: direction = "favors female applicants"
else:         direction = "shows no gender gap"

if   abs(gap) < 2:  severity = "negligible"
elif abs(gap) < 5:  severity = "modest"
elif abs(gap) < 10: severity = "notable"
elif abs(gap) < 15: severity = "significant"
else:               severity = "severe"

report = f"""
{SEP}
   AI HIRING BIAS REPORT — GENDER ANALYSIS (1000 CANDIDATES)
{SEP}

  Total candidates tested : {total:,}

  MALE CANDIDATES ({len(male_df):,} total)
    Hired     : {m_hired:,}  ({m_rate:.1f}%)
    Not hired : {len(male_df)-m_hired:,}  ({100-m_rate:.1f}%)

  FEMALE CANDIDATES ({len(female_df):,} total)
    Hired     : {f_hired:,}  ({f_rate:.1f}%)
    Not hired : {len(female_df)-f_hired:,}  ({100-f_rate:.1f}%)

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
with open("results_summary_1000.txt", "w") as fh:
    fh.write(report)
print("  Report saved to: results_summary_1000.txt\n")

# ── Charts ────────────────────────────────────────────────────────────────────
overall = pd.DataFrame({"Male": [m_rate], "Female": [f_rate]}, index=["All Candidates"])
bar_chart(overall,
          title="AI Hiring Model — Hire Rate by Gender (1000 Candidates)",
          xlabel="", filename="bias_gender_chart_1000.png", figsize=(7, 6), bar_width=0.2)

field_pivot = hire_rate_pivot(df, "Job").reindex(sorted(df["Job"].unique()))
bar_chart(field_pivot,
          title="Gender Bias by Job Field — 1000 Candidates\n(Hire Rate Comparison — Male vs Female)",
          xlabel="Job Field", filename="bias_by_field_1000.png",
          figsize=(12, 7), rotate=True)

tier_order = ["Low  (Vocational/\nAssociate's)", "Mid  (Bachelor's)", "High  (Master's+)"]
tier_pivot = hire_rate_pivot(df, "Uni_Tier")
tier_pivot = tier_pivot.reindex([t for t in tier_order if t in tier_pivot.index])
bar_chart(tier_pivot,
          title="Gender Bias by Education Tier — 1000 Candidates\n(Hire Rate Comparison — Male vs Female)",
          xlabel="Education Tier", filename="bias_by_university_tier_1000.png", figsize=(10, 7))

print("\nAll done.\n")
