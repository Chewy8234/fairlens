"""
regenerate_all.py
Regenerates test_applicants.csv (200), test_applicants_500.csv (500),
test_applicants_1000.csv (1000) with a balanced 50/50 gender split
across all job fields. model.pkl makes every single hiring decision.
Does NOT touch gender_biased_final.csv or model.pkl.
"""
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

ETHNICITIES = ["AF", "BA", "H", "EA", "WE", "WA"]
CITIES      = ["New York", "Chicago", "Houston", "Los Angeles", "Seattle",
               "Boston", "Atlanta", "Denver", "Miami", "Phoenix"]
SALONS      = ["Trendy Cuts Salon", "Elite Style Studio", "The Hair Lounge",
               "Urban Chic Salon", "Prestige Hair Co", "Studio One Salon"]
HOSPITALS   = ["City General Hospital", "Metro Medical Center", "St. Luke's Hospital",
               "University Hospital", "Riverside Clinic", "Memorial Medical Center"]
COMPANIES   = ["TechCorp", "BuildRight Inc", "Apex Solutions", "Metro Group",
               "Nexus Industries", "Summit Engineering", "Horizon Consulting"]

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

EDUCATION = {
    "Hairdresser": [
        "{name} completed a cosmetology degree at the {city} School of Cosmetology.",
        "{name} holds an Associate Degree in Cosmetology from {city} Beauty College.",
        "{name} earned a Certificate in Hairdressing from the {city} Institute of Beauty.",
        "{name} graduated with a Diploma in Barbering and Styling from {city} Vocational School.",
    ],
    "Engineer": [
        "{name} holds a Bachelor's in Mechanical Engineering from {city} State University.",
        "{name} earned a Master's in Electrical Engineering from {city} Tech Institute.",
        "{name} completed a Bachelor's in Civil Engineering from the University of {city}.",
        "{name} holds a PhD in Computer Engineering from {city} Institute of Technology.",
        "{name} graduated with a Bachelor's in Software Engineering from {city} Polytechnic.",
    ],
    "Nurse": [
        "{name} holds a Bachelor's in Nursing from {city} University Medical School.",
        "{name} earned an Associate Degree in Nursing from {city} Community College.",
        "{name} completed a Master's in Nursing Practice from {city} Health Sciences University.",
        "{name} holds a BSN from {city} General Hospital School of Nursing.",
    ],
    "Secretary": [
        "{name} holds an Associate Degree in Business Administration from {city} Community College.",
        "{name} earned a Bachelor's in Office Management from {city} State College.",
        "{name} completed a Certificate in Administrative Support from {city} Business Institute.",
        "{name} holds a Bachelor's in Communications from {city} University.",
    ],
    "Surgeon": [
        "{name} holds an MD from {city} School of Medicine.",
        "{name} earned a PhD in Surgical Sciences from {city} Medical University.",
        "{name} completed a Master's in Surgery from {city} General Hospital Medical School.",
        "{name} holds an MD with residency training at {city} Academic Medical Center.",
    ],
    "Construction Worker": [
        "{name} holds a Certificate in Construction Technology from {city} Vocational Institute.",
        "{name} earned an Associate Degree in Building Trades from {city} Technical College.",
        "{name} completed a Bachelor's in Construction Management from {city} State University.",
        "{name} holds an Apprenticeship Certificate in Carpentry from {city} Trades Council.",
    ],
}
WORK_EXP = {
    "Hairdresser": [
        "{name} has {n} years of experience at {salon}, specializing in precision cuts and advanced color techniques.",
        "{name} worked for {n} years as a lead stylist at {salon}, recognized for client satisfaction and creativity.",
        "{name} brings {n} years of barbering and styling expertise from {salon}, excelling in modern and classic techniques.",
    ],
    "Engineer": [
        "{name} has {n} years of engineering experience at {company}, leading cross-functional technical projects.",
        "{name} spent {n} years at {company} designing mechanical systems and overseeing product development cycles.",
        "{name} brings {n} years of software development experience from {company}, building scalable infrastructure solutions.",
    ],
    "Nurse": [
        "{name} has {n} years of clinical nursing experience at {hospital}, providing compassionate patient-centered care.",
        "{name} worked for {n} years in the ICU at {hospital}, specializing in critical care and emergency response.",
        "{name} brings {n} years of pediatric nursing experience from {hospital}, known for exceptional family communication.",
    ],
    "Secretary": [
        "{name} has {n} years of administrative experience supporting C-suite executives at {company}.",
        "{name} managed all office operations for {n} years at {company}, coordinating calendars, travel, and communications.",
        "{name} brings {n} years of executive assistant experience from {company}, handling confidential correspondence.",
    ],
    "Surgeon": [
        "{name} has {n} years of surgical experience at {hospital}, performing complex minimally invasive procedures.",
        "{name} completed {n} years of residency and surgical practice at {hospital}, specializing in orthopedic surgery.",
        "{name} brings {n} years of cardiac surgery expertise from {hospital}, with an outstanding patient outcome record.",
    ],
    "Construction Worker": [
        "{name} has {n} years of hands-on construction experience at {company}, skilled in structural and finishing work.",
        "{name} worked on major infrastructure projects for {n} years with {company}, operating heavy equipment safely.",
        "{name} brings {n} years of residential and commercial construction experience from {company}.",
    ],
}
SKILLS = {
    "Hairdresser": [
        "Proficient in cutting, coloring, highlighting, and styling. Strong customer service and interpersonal skills.",
        "Skilled in balayage, keratin treatments, and precision cuts. Excellent time management in fast-paced salon environments.",
        "Expert in texturizing, men's grooming, and advanced color techniques. Strong problem-solving and attention to detail.",
    ],
    "Engineer": [
        "Skilled in CAD, project management, and systems design. Proficient in Python, MATLAB, and AutoCAD.",
        "Expert in machine learning, data pipelines, and cloud infrastructure. Strong analytical and problem-solving skills.",
        "Proficient in structural analysis, finite element modeling, and technical documentation. Strong leadership under pressure.",
    ],
    "Nurse": [
        "Expert in patient assessment, IV therapy, and medication management. Strong empathy and crisis communication.",
        "Skilled in wound care, patient education, and electronic health records. Calm under pressure with excellent bedside manner.",
        "Proficient in critical care protocols, triage assessment, and family counseling. Strong team collaboration skills.",
    ],
    "Secretary": [
        "Proficient in MS Office Suite, scheduling software, and correspondence drafting. Excellent multitasking and discretion.",
        "Skilled in calendar management, travel coordination, and meeting facilitation. Fast typing with superior attention to detail.",
        "Expert in office administration, document management, and executive support. Strong organizational and communication skills.",
    ],
    "Surgeon": [
        "Expert in laparoscopic and robotic surgical techniques. Strong decision-making under pressure and exceptional dexterity.",
        "Skilled in trauma surgery and post-operative care. Proficient in surgical simulation and resident teaching.",
        "Proficient in minimally invasive cardiac procedures and surgical planning. Outstanding precision and clinical judgment.",
    ],
    "Construction Worker": [
        "Skilled in carpentry, concrete work, scaffolding, and equipment operation. Strong safety awareness and physical endurance.",
        "Proficient in blueprint reading, site coordination, and OSHA compliance. Reliable team player with strong work ethic.",
        "Expert in structural framing, masonry, and project scheduling. Strong mechanical aptitude and attention to safety standards.",
    ],
}
AWARDS = {
    "Hairdresser": [
        "{name} received the 'Best Stylist' award at the {city} Hair and Beauty Expo.",
        "{name} was named Top Hairdresser of the Year by {salon}.",
        "{name} was recognized as an Outstanding Stylist at the {city} Hair Expo for exceptional talent.",
    ],
    "Engineer": [
        "{name} received the 'Innovation Award' from the {city} Engineering Society.",
        "{name} was recognized as Engineer of the Quarter at {company} for outstanding project delivery.",
        "{name} earned the 'Excellence in Engineering' commendation from {company} for leading a critical project.",
    ],
    "Nurse": [
        "{name} received the 'Excellence in Patient Care' award from {hospital}.",
        "{name} was named Nurse of the Year at {hospital} for exceptional dedication.",
        "{name} received the 'Compassionate Care' recognition from {hospital} for outstanding patient advocacy.",
    ],
    "Secretary": [
        "{name} was awarded 'Administrative Professional of the Year' at {company}.",
        "{name} received the 'Outstanding Support Staff' recognition at the {city} Business Excellence Gala.",
        "{name} was commended for exceptional organizational efficiency at {company}'s annual awards ceremony.",
    ],
    "Surgeon": [
        "{name} received the 'Outstanding Surgeon' award from the {city} Medical Association.",
        "{name} was recognized for surgical excellence and innovation at {hospital}'s annual awards ceremony.",
        "{name} received the 'Surgical Innovation Award' from {hospital} for pioneering a minimally invasive technique.",
    ],
    "Construction Worker": [
        "{name} received the 'Safety Excellence' award from {company} for zero-incident project leadership.",
        "{name} was recognized as Top Performer on the {city} Downtown Infrastructure Project.",
        "{name} earned the 'Outstanding Craftsman' award from {company} for exceptional quality on-site.",
    ],
}

MALE_COLOR   = "#2E86C1"
FEMALE_COLOR = "#E74C3C"
SEP = "=" * 60


def pick(rng, lst):
    return lst[rng.integers(len(lst))]


def edu_tier(text):
    if not isinstance(text, str):
        return "Unknown"
    t = text.lower()
    if any(k in t for k in ["master", "mba", "phd", "doctorate", "md ", "doctor of"]):
        return "High  (Master's+)"
    if any(k in t for k in ["bachelor", "b.s.", "b.a.", "bsn", "undergraduate"]):
        return "Mid  (Bachelor's)"
    return "Low  (Vocational/\nAssociate's)"


def make_person(rng, gender, job):
    city  = pick(rng, CITIES)
    eth   = pick(rng, ETHNICITIES)
    gpa   = round(float(rng.uniform(2.4, 4.0)), 1)
    yrs   = int(rng.integers(1, 16))
    first = pick(rng, MALE_FIRST if gender == "male" else FEMALE_FIRST)
    last  = pick(rng, MALE_LAST  if gender == "male" else FEMALE_LAST)
    name  = f"{first} {last}"
    email = f"{first.lower()}.{last.lower().replace(chr(39), '')}@email.com"
    edu   = pick(rng, EDUCATION[job]).format(name=first, city=city)
    exp   = pick(rng, WORK_EXP[job]).format(
                name=first, n=yrs,
                salon=pick(rng, SALONS),
                hospital=pick(rng, HOSPITALS),
                company=pick(rng, COMPANIES))
    skill = pick(rng, SKILLS[job])
    award = pick(rng, AWARDS[job]).format(
                name=first, city=city,
                salon=pick(rng, SALONS),
                hospital=pick(rng, HOSPITALS),
                company=pick(rng, COMPANIES))
    return {"Name": name, "Gender": gender, "Email": email, "Ethnicity": eth,
            "Education": edu, "GPA": str(gpa), "Work Experience": exp,
            "Skills": skill, "Awards": award, "Fit": "", "Job": job}


def generate_csv(total, seed, filename, jobs):
    """
    Distributes `total` candidates evenly across all job fields with a
    balanced 50/50 gender split per field. model.pkl makes every single
    hiring decision — no outcomes are hardcoded.
    """
    rng    = np.random.default_rng(seed)
    n_jobs = len(jobs)
    base   = total // n_jobs
    extras = total % n_jobs
    job_counts = {j: base + (1 if i < extras else 0) for i, j in enumerate(jobs)}

    rows = []
    for job, n in job_counts.items():
        n_female = n // 2
        n_male   = n - n_female
        genders  = ["female"] * n_female + ["male"] * n_male
        rng.shuffle(genders)
        for gender in genders:
            rows.append(make_person(rng, gender, job))

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    if os.path.exists(filename):
        os.remove(filename)
    df.to_csv(filename, index=False)

    g = df["Gender"].str.lower().value_counts()
    print(f"\n  Generated {filename}  ({len(df)} rows — "
          f"{g.get('male', 0)} male / {g.get('female', 0)} female)")
    print(f"  {'Job':<22} {'Total':>6}  {'Male':>6}  {'Female':>7}  {'F%':>6}")
    print(f"  {'-'*55}")
    for job, n in job_counts.items():
        jdf = df[df["Job"] == job]
        n_m = (jdf["Gender"].str.lower() == "male").sum()
        n_f = (jdf["Gender"].str.lower() == "female").sum()
        f_pct = n_f / n * 100
        print(f"  {job:<22} {n:>6}  {n_m:>6}  {n_f:>7}  {f_pct:>5.1f}%")
    return df


def run_model(df, clf, tfidf, gender_le, ethnicity_le, job_le, num_cols, **_):
    df = df.copy().map(lambda x: x.strip() if isinstance(x, str) else x)
    df["GPA_num"]       = pd.to_numeric(df["GPA"], errors="coerce").fillna(0)
    df["Gender_enc"]    = gender_le.transform(df["Gender"])
    df["Ethnicity_enc"] = ethnicity_le.transform(df["Ethnicity"])
    df["Job_enc"]       = job_le.transform(df["Job"])
    df["text_combined"] = (df["Education"].fillna("") + " " +
                           df["Work Experience"].fillna("") + " " +
                           df["Skills"].fillna("") + " " +
                           df["Awards"].fillna(""))
    X = hstack([csr_matrix(df[num_cols].values),
                tfidf.transform(df["text_combined"])])
    df["Predicted_Fit"] = clf.predict(X)
    df["Hired"]         = df["Predicted_Fit"].isin(["good", "very good"]).astype(int)
    df["Gender"]        = df["Gender"].str.title()
    df["Uni_Tier"]      = df["Education"].apply(edu_tier)
    return df


def print_stats(df, label, txt_file):
    total     = len(df)
    hired     = df["Hired"].sum()
    male_df   = df[df["Gender"] == "Male"]
    female_df = df[df["Gender"] == "Female"]
    m_hired   = male_df["Hired"].sum()
    f_hired   = female_df["Hired"].sum()
    m_rate    = m_hired / len(male_df)   * 100
    f_rate    = f_hired / len(female_df) * 100
    gap       = m_rate - f_rate
    severity  = ("negligible"   if abs(gap) < 2  else "modest"       if abs(gap) < 5
                 else "notable" if abs(gap) < 10  else "significant"  if abs(gap) < 15
                 else "severe")
    direction = ("favors male applicants"    if gap > 0
                 else "favors female applicants" if gap < 0
                 else "shows no gender gap")
    report = (
        f"\n{SEP}\n"
        f"   AI HIRING BIAS REPORT — GENDER ANALYSIS ({label})\n"
        f"{SEP}\n\n"
        f"  Total candidates : {total:,}\n"
        f"  Hired            : {hired:,}  ({hired/total*100:.1f}%)\n"
        f"  Not hired        : {total-hired:,}  ({(total-hired)/total*100:.1f}%)\n\n"
        f"  MALE CANDIDATES ({len(male_df):,} total)\n"
        f"    Hired     : {m_hired:,}  ({m_rate:.1f}%)\n"
        f"    Not hired : {len(male_df)-m_hired:,}  ({100-m_rate:.1f}%)\n\n"
        f"  FEMALE CANDIDATES ({len(female_df):,} total)\n"
        f"    Hired     : {f_hired:,}  ({f_rate:.1f}%)\n"
        f"    Not hired : {len(female_df)-f_hired:,}  ({100-f_rate:.1f}%)\n\n"
        f"  GENDER BIAS GAP : {gap:+.1f} percentage points\n"
        f"  (Male hire rate minus Female hire rate)\n\n"
        f"  Severity : {severity} — {direction}\n"
        f"{SEP}\n"
    )
    print(report)
    with open(txt_file, "w") as fh:
        fh.write(report)
    print(f"  Report saved → {txt_file}\n")


def hire_rate_pivot(df, group_col):
    tbl = (df.groupby([group_col, "Gender"])["Hired"]
             .agg(hired="sum", total="count").reset_index())
    tbl["rate"] = tbl["hired"] / tbl["total"] * 100
    return tbl.pivot(index=group_col, columns="Gender", values="rate")


def bar_chart(pivot, title, xlabel, filename, figsize=(10, 6), rotate=False, bar_width=0.35):
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")
    groups      = pivot.index.tolist()
    x           = np.arange(len(groups))
    width       = bar_width
    male_vals   = pivot["Male"].fillna(0).values   if "Male"   in pivot.columns else np.zeros(len(groups))
    female_vals = pivot["Female"].fillna(0).values if "Female" in pivot.columns else np.zeros(len(groups))

    ax.bar(x - width/2, male_vals,   width, color=MALE_COLOR,   edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + width/2, female_vals, width, color=FEMALE_COLOR, edgecolor="white", linewidth=0.8, zorder=3)

    for val, xpos in zip(male_vals, x - width/2):
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
    ax.set_ylim(0, min(100, max(male_vals.max(), female_vals.max()) + 14))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[mpatches.Patch(color=MALE_COLOR,   label="Male"),
                        mpatches.Patch(color=FEMALE_COLOR, label="Female")],
              fontsize=11, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def make_charts(df, suffix, label):
    male_df   = df[df["Gender"] == "Male"]
    female_df = df[df["Gender"] == "Female"]
    overall   = pd.DataFrame(
        {"Male":   [male_df["Hired"].mean()   * 100],
         "Female": [female_df["Hired"].mean() * 100]},
        index=["All Candidates"])
    bar_chart(overall,
              title=f"AI Hiring Model — Hire Rate by Gender ({label})",
              xlabel="", filename=f"bias_gender_chart_{suffix}.png", figsize=(7, 6),
              bar_width=0.07)

    field_pivot = hire_rate_pivot(df, "Job").reindex(sorted(df["Job"].unique()))
    bar_chart(field_pivot,
              title=f"Gender Bias by Job Field — {label}\n(Hire Rate Comparison — Male vs Female)",
              xlabel="Job Field", filename=f"bias_by_field_{suffix}.png",
              figsize=(12, 7), rotate=True)

    tier_order = ["Low  (Vocational/\nAssociate's)", "Mid  (Bachelor's)", "High  (Master's+)"]
    tier_pivot = hire_rate_pivot(df, "Uni_Tier")
    tier_pivot = tier_pivot.reindex([t for t in tier_order if t in tier_pivot.index])
    bar_chart(tier_pivot,
              title=f"Gender Bias by Education Tier — {label}\n(Hire Rate Comparison — Male vs Female)",
              xlabel="Education Tier", filename=f"bias_by_university_tier_{suffix}.png",
              figsize=(10, 7))


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  Loading hiring_model.pkl ...\n{SEP}")
with open("hiring_model.pkl", "rb") as f:
    bundle = pickle.load(f)
clf      = bundle["model"]
tfidf    = bundle["tfidf"]
num_cols = bundle["num_cols"]

train        = pd.read_csv("ai-hiring-model/gender_biased_final.csv")
train        = train.map(lambda x: x.strip() if isinstance(x, str) else x)
jobs         = sorted(train["Job"].unique())   # read fields directly from training data
gender_le    = LabelEncoder().fit(train["Gender"])
ethnicity_le = LabelEncoder().fit(train["Ethnicity"])
job_le       = LabelEncoder().fit(train["Job"])
print(f"  Model loaded.  Job fields found in gender_biased_final.csv: {jobs}\n")

print(f"{SEP}\n  STEP 1 — GENERATING CSV FILES (balanced 50/50 gender split)\n{SEP}")
df200  = generate_csv(200,  seed=1,  filename="test_applicants.csv",     jobs=jobs)
df500  = generate_csv(500,  seed=42, filename="test_applicants_500.csv", jobs=jobs)
df1000 = generate_csv(1000, seed=99, filename="test_applicants_1000.csv",jobs=jobs)

print(f"\n{SEP}\n  STEP 2 — RUNNING MODEL\n{SEP}")
run_args = (clf, tfidf, gender_le, ethnicity_le, job_le, num_cols)
df200  = run_model(df200,  *run_args)
df500  = run_model(df500,  *run_args)
df1000 = run_model(df1000, *run_args)
print("  Done.\n")

print(f"{SEP}\n  STEP 3 — TERMINAL RESULTS\n{SEP}")
print_stats(df200,  "200 CANDIDATES",  "results_summary_200.txt")
print_stats(df500,  "500 CANDIDATES",  "results_summary_500.txt")
print_stats(df1000, "1000 CANDIDATES", "results_summary_1000.txt")

print(f"{SEP}\n  STEP 4 — SAVING 9 CHARTS\n{SEP}")
make_charts(df200,  suffix="200",  label="200 Candidates")
make_charts(df500,  suffix="500",  label="500 Candidates")
make_charts(df1000, suffix="1000", label="1000 Candidates")

print(f"\n{SEP}\n  ALL DONE\n{SEP}\n")
