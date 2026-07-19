import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

# ── Fictional name pools ──────────────────────────────────────────────────────
MALE_NAMES = [
    "James Carter", "Michael Brown", "David Kim", "Robert Chen", "William Davis",
    "Daniel Martinez", "Matthew Johnson", "Christopher Lee", "Andrew Wilson",
    "Joshua Thompson", "Ryan Anderson", "Justin Taylor", "Brandon Harris",
    "Tyler Robinson", "Nathan Clark", "Kevin Lewis", "Brian Walker", "Jason Hall",
    "Eric Young", "Adam Allen", "Sean Wright", "Patrick Scott", "Timothy Green",
    "Gregory Adams", "Jonathan Baker", "Steven Nelson", "Kyle Mitchell",
    "Aaron Perez", "Jeremy Roberts", "Zachary Turner", "Derek White", "Marcus Hill",
    "Darius Jackson", "Malik Washington", "Jamal Brooks", "DeShawn Coleman",
    "Terrence Hughes", "Dominique Foster", "Antoine Powell", "Jerome Patterson",
    "Carlos Rivera", "Miguel Flores", "Diego Ramirez", "Eduardo Ortiz",
    "Luis Torres", "Marco Vargas", "Pedro Gonzalez", "Alejandro Reyes",
    "Wei Zhang", "Jian Liu", "Ming Wang", "Hao Chen", "Ravi Patel",
    "Amir Hassan", "Khalid Rahman", "Omar Abdullah", "Yusuf Al-Rashid",
    "Liam O'Brien", "Ethan Murphy", "Noah Walsh", "Finn McCarthy",
    "Oliver Bennett", "George Thornton", "Harry Whitfield", "Charlie Ashford",
    "Lars Eriksson", "Henrik Lindqvist", "Matteo Romano", "Luca Ferrari",
    "Hiroshi Tanaka", "Kenji Yamamoto", "Soren Andersen", "Pieter van der Berg",
    "Andrei Popescu", "Bogdan Ionescu", "Viktor Novak", "Dmitri Volkov",
    "Ibrahim Diallo", "Kwame Mensah", "Chukwudi Okafor", "Emeka Nwosu",
    "Samuel Osei", "Kofi Asante", "Tendai Moyo", "Sipho Ndlovu",
    "Arjun Sharma", "Vikram Singh", "Rahul Gupta", "Suresh Nair",
    "Yosef Ben-David", "Amit Cohen", "Oren Shapiro", "Natan Levy",
    "Takeshi Nakamura", "Sho Watanabe", "Jun Hayashi", "Ren Kobayashi",
    "Chen Wei", "Bao Nguyen", "Minh Tran", "Duc Pham",
]

FEMALE_NAMES = [
    "Emily Johnson", "Jessica Williams", "Sarah Davis", "Ashley Brown",
    "Amanda Wilson", "Stephanie Moore", "Melissa Taylor", "Rebecca Anderson",
    "Lauren Thomas", "Rachel Jackson", "Megan White", "Brittany Harris",
    "Kayla Martin", "Amber Thompson", "Danielle Garcia", "Crystal Martinez",
    "Shannon Robinson", "Tiffany Clark", "Jennifer Lewis", "Christina Hall",
    "Olivia Young", "Emma Walker", "Ava Allen", "Isabella Wright",
    "Sophia Scott", "Mia Green", "Charlotte Adams", "Amelia Baker",
    "Harper Nelson", "Evelyn Mitchell", "Abigail Perez", "Elizabeth Roberts",
    "Aaliyah Washington", "Jasmine Brooks", "Keisha Coleman", "Latisha Hughes",
    "Shaniqua Foster", "Tamara Powell", "Monique Patterson", "Deja Jefferson",
    "Maria Rodriguez", "Sofia Hernandez", "Isabella Lopez", "Valentina Diaz",
    "Camila Perez", "Daniela Cruz", "Fernanda Morales", "Natalia Reyes",
    "Mei Lin", "Xiao Wang", "Yu Chen", "Ling Zhang", "Priya Patel",
    "Ananya Sharma", "Divya Singh", "Kavya Nair", "Fatima Al-Hassan",
    "Layla Rahman", "Nour Abdullah", "Yasmin Khalid", "Siobhan O'Connor",
    "Aoife Murphy", "Niamh Walsh", "Roisin Brennan", "Freya Larsson",
    "Astrid Lindqvist", "Chiara Romano", "Giulia Ferrari", "Valentina Rossi",
    "Yuki Tanaka", "Sakura Yamamoto", "Hana Nakamura", "Emi Watanabe",
    "Nadia Popescu", "Elena Ionescu", "Katarzyna Kowalski", "Agnieszka Nowak",
    "Amara Diallo", "Ama Mensah", "Ngozi Okafor", "Adaeze Nwosu",
    "Abena Asante", "Akosua Osei", "Rudo Moyo", "Thandi Dlamini",
    "Leila Ahmadi", "Shirin Hosseini", "Zara Aziz", "Nadia Malik",
    "Seo-Yeon Park", "Ji-Young Kim", "Min-Ji Lee", "Ye-Jin Choi",
    "Linh Nguyen", "Phuong Tran", "An Pham", "Thu Bui",
    "Nkechi Eze", "Chidinma Obi", "Ifeoma Chukwu", "Adaora Onyeka",
]

JOBS = ["Hairdresser", "Engineer", "Nurse", "Secretary", "Surgeon", "Construction Worker"]
ETHNICITIES = ["AF", "BA", "H", "EA", "WE", "WA"]

EDU_TEMPLATES = {
    "Hairdresser": [
        ("Associate Degree in Cosmetology from {city} Beauty College", 2),
        ("Bachelor's in Cosmetology from {city} School of Cosmetology", 3),
        ("Certificate in Hairdressing from {city} Institute of Beauty", 2),
    ],
    "Engineer": [
        ("Bachelor's in Mechanical Engineering from {city} State University", 3),
        ("Master's in Electrical Engineering from {city} Tech", 4),
        ("Bachelor's in Civil Engineering from University of {city}", 3),
        ("PhD in Computer Engineering from {city} Institute of Technology", 5),
    ],
    "Nurse": [
        ("Bachelor's in Nursing from {city} University Medical School", 3),
        ("Associate Degree in Nursing from {city} Community College", 2),
        ("Master's in Nursing Practice from {city} Health Sciences University", 4),
    ],
    "Secretary": [
        ("Associate Degree in Business Administration from {city} Community College", 2),
        ("Bachelor's in Office Management from {city} State College", 3),
        ("Certificate in Administrative Support from {city} Business Institute", 2),
    ],
    "Surgeon": [
        ("MD from {city} School of Medicine", 5),
        ("PhD in Surgical Sciences from {city} Medical University", 5),
        ("Master's in Surgery from {city} General Hospital Medical School", 4),
    ],
    "Construction Worker": [
        ("Certificate in Construction Technology from {city} Vocational Institute", 2),
        ("Associate Degree in Building Trades from {city} Technical College", 2),
        ("Bachelor's in Construction Management from {city} State University", 3),
    ],
}

EXP_TEMPLATES = {
    "Hairdresser": [
        "{name} has {n} years of experience at {salon}, specializing in precision cuts and color techniques.",
        "{name} worked for {n} years as a stylist at {salon}, known for attention to detail.",
    ],
    "Engineer": [
        "{name} has {n} years of engineering experience at {company}, leading technical projects.",
        "{name} spent {n} years at {company} developing software systems and managing cross-functional teams.",
    ],
    "Nurse": [
        "{name} has {n} years of nursing experience at {hospital}, providing patient-centered care.",
        "{name} worked for {n} years in the ICU at {hospital}, specializing in critical care.",
    ],
    "Secretary": [
        "{name} has {n} years of administrative experience supporting executives at {company}.",
        "{name} managed office operations for {n} years at {company}, coordinating schedules and communications.",
    ],
    "Surgeon": [
        "{name} has {n} years of surgical experience at {hospital}, performing complex procedures.",
        "{name} completed {n} years of residency and practice at {hospital}, specializing in minimally invasive surgery.",
    ],
    "Construction Worker": [
        "{name} has {n} years of hands-on construction experience at {company}, skilled in structural work.",
        "{name} worked on major infrastructure projects for {n} years with {company}.",
    ],
}

SKILLS_TEMPLATES = {
    "Hairdresser": "Proficient in cutting, coloring, and styling. Strong customer service and communication skills.",
    "Engineer": "Skilled in CAD, project management, and problem-solving. Proficient in Python and MATLAB.",
    "Nurse": "Expertise in patient assessment, medication administration, and critical care. Strong empathy and communication.",
    "Secretary": "Proficient in MS Office, scheduling, and correspondence. Excellent organizational and multitasking skills.",
    "Surgeon": "Expert in laparoscopic and open surgical techniques. Strong analytical skills and attention to detail.",
    "Construction Worker": "Skilled in carpentry, concrete work, and equipment operation. Strong physical endurance and safety awareness.",
}

AWARDS_TEMPLATES = {
    "Hairdresser": ["{name} received the 'Best Stylist' award at the {city} Hair Expo.", "Recognized as a top hairdresser at the {city} Beauty Awards."],
    "Engineer": ["{name} received the 'Innovation Award' from {city} Engineering Society.", "Recognized as Engineer of the Year by {company}."],
    "Nurse": ["{name} received the 'Excellence in Patient Care' award from {hospital}.", "Recognized as Nurse of the Month at {hospital}."],
    "Secretary": ["{name} was awarded 'Administrative Professional of the Year' at {company}.", "Recognized for outstanding support at the {city} Business Excellence Awards."],
    "Surgeon": ["{name} received the 'Outstanding Surgeon' award from the {city} Medical Association.", "Recognized for excellence in surgical innovation at {hospital}."],
    "Construction Worker": ["{name} received the 'Safety Excellence' award from {company}.", "Recognized as top performer on the {city} Infrastructure Project."],
}

CITIES = ["New York", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio",
          "San Diego", "Dallas", "San Jose", "Austin", "Los Angeles", "Seattle"]
SALONS = ["Trendy Cuts Salon", "Elite Style Studio", "The Hair Lounge", "Urban Chic Salon"]
HOSPITALS = ["City General Hospital", "Metro Medical Center", "St. Luke's Hospital", "University Hospital"]
COMPANIES = ["TechCorp", "BuildRight Construction", "Apex Solutions", "Metro Group", "Nexus Industries"]


def pick(lst):
    return lst[np.random.randint(len(lst))]


def generate_individual(idx, gender):
    job = pick(JOBS)
    city = pick(CITIES)
    first_name = pick(MALE_NAMES if gender == "male" else FEMALE_NAMES).split()[0]
    full_name = pick(MALE_NAMES if gender == "male" else FEMALE_NAMES)
    ethnicity = pick(ETHNICITIES)
    gpa = round(np.random.uniform(2.5, 4.0), 1)
    exp_years = np.random.randint(1, 15)

    edu_template, edu_rank = pick(EDU_TEMPLATES[job])
    education = edu_template.format(city=city)

    exp_template = pick(EXP_TEMPLATES[job])
    work_exp = exp_template.format(
        name=first_name, n=exp_years,
        salon=pick(SALONS), hospital=pick(HOSPITALS), company=pick(COMPANIES)
    )

    skills = SKILLS_TEMPLATES[job]

    award_template = pick(AWARDS_TEMPLATES[job])
    awards = award_template.format(
        name=first_name, city=city,
        hospital=pick(HOSPITALS), company=pick(COMPANIES)
    )

    email = f"{full_name.lower().replace(' ', '.')}@email.com"

    return {
        "Name": full_name,
        "Gender": gender,
        "Email": email,
        "Ethnicity": ethnicity,
        "Education": education,
        "GPA": str(gpa),
        "Work Experience": work_exp,
        "Skills": skills,
        "Awards": awards,
        "Fit": "",   # to be predicted
        "Job": job,
    }


# ── Generate 200 candidates (100 male, 100 female) ───────────────────────────
records = []
for i in range(100):
    records.append(generate_individual(i, "male"))
for i in range(100):
    records.append(generate_individual(i + 100, "female"))

candidates = pd.DataFrame(records)
candidates = candidates.sample(frac=1, random_state=42).reset_index(drop=True)
candidates.to_csv("test_candidates_200.csv", index=False)
print(f"Generated 200 candidates → test_candidates_200.csv")

# ── Load model bundle ─────────────────────────────────────────────────────────
with open("hiring_model.pkl", "rb") as f:
    bundle = pickle.load(f)

clf = bundle["model"]
tfidf = bundle["tfidf"]

# ── Encode features (same logic as training) ─────────────────────────────────
df_train = pd.read_csv("ai-hiring-model/gender_biased_final.csv")
df_train = df_train.map(lambda x: x.strip() if isinstance(x, str) else x)

gender_le = LabelEncoder().fit(df_train["Gender"])
ethnicity_le = LabelEncoder().fit(df_train["Ethnicity"])
job_le = LabelEncoder().fit(df_train["Job"])

test = candidates.copy()
test["GPA_num"] = pd.to_numeric(test["GPA"], errors="coerce").fillna(0)
test["Gender_enc"] = gender_le.transform(test["Gender"])
test["Ethnicity_enc"] = ethnicity_le.transform(test["Ethnicity"])
test["Job_enc"] = job_le.transform(test["Job"])
test["text_combined"] = (
    test["Education"].fillna("") + " " +
    test["Work Experience"].fillna("") + " " +
    test["Skills"].fillna("") + " " +
    test["Awards"].fillna("")
)

from scipy.sparse import hstack, csr_matrix

num_cols = ["Gender_enc", "Ethnicity_enc", "Job_enc", "GPA_num"]
X_num = csr_matrix(test[num_cols].values)
X_tfidf = tfidf.transform(test["text_combined"])
X_final = hstack([X_num, X_tfidf])

# ── Predict ───────────────────────────────────────────────────────────────────
predictions = clf.predict(X_final)
test["Predicted_Fit"] = predictions
test["Hired"] = test["Predicted_Fit"].isin(["good", "very good"])

# Save predictions
out = test[["Name", "Gender", "Ethnicity", "Job", "GPA", "Predicted_Fit", "Hired"]]
out.to_csv("predictions_200.csv", index=False)
print("Predictions saved → predictions_200.csv\n")

# ── Console analysis ──────────────────────────────────────────────────────────
total = len(test)
hired = test["Hired"].sum()
not_hired = total - hired

print("=" * 55)
print("         HIRING PREDICTION RESULTS — 200 CANDIDATES")
print("=" * 55)
print(f"\n  Total candidates :  {total}")
print(f"  Hired            :  {hired}  ({hired/total*100:.1f}%)")
print(f"  Not hired        :  {not_hired}  ({not_hired/total*100:.1f}%)")

print("\n── By Gender ──────────────────────────────────────────")
gender_summary = test.groupby("Gender").agg(
    Total=("Hired", "count"),
    Hired=("Hired", "sum")
).assign(Hire_Rate=lambda d: (d["Hired"] / d["Total"] * 100).round(1))
print(gender_summary.to_string())

male_rate = gender_summary.loc["male", "Hire_Rate"]
female_rate = gender_summary.loc["female", "Hire_Rate"]
gap = male_rate - female_rate
print(f"\n  Gender hire-rate gap (male − female): {gap:+.1f} pp")
if abs(gap) >= 10:
    print("  ⚠  SIGNIFICANT gender bias detected (≥10 pp difference).")
elif abs(gap) >= 5:
    print("  ⚠  MODERATE gender bias detected (5–10 pp difference).")
else:
    print("  ✓  No significant gender bias detected (<5 pp difference).")

print("\n── Predicted Fit Distribution ─────────────────────────")
fit_dist = test.groupby(["Gender", "Predicted_Fit"]).size().unstack(fill_value=0)
print(fit_dist.to_string())

print("\n── Hire Rate by Job × Gender ──────────────────────────")
job_gender = test.groupby(["Job", "Gender"]).agg(
    Total=("Hired", "count"),
    Hired=("Hired", "sum")
).assign(Rate=lambda d: (d["Hired"] / d["Total"] * 100).round(1))
print(job_gender.to_string())

# ── Visualisations ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("AI Hiring Model — 200 Candidate Analysis", fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

palette = {"male": "#4C72B0", "female": "#DD8452"}

# 1. Hired vs Not Hired (pie)
ax1 = fig.add_subplot(gs[0, 0])
ax1.pie([hired, not_hired], labels=["Hired", "Not Hired"],
        colors=["#2ca02c", "#d62728"], autopct="%1.1f%%", startangle=90)
ax1.set_title("Overall Hired vs Not Hired")

# 2. Hired count by gender (bar)
ax2 = fig.add_subplot(gs[0, 1])
gender_hired = test.groupby("Gender")["Hired"].sum().reset_index()
sns.barplot(data=gender_hired, x="Gender", y="Hired", palette=palette, ax=ax2)
ax2.set_title("Total Hired by Gender")
ax2.set_ylabel("# Hired")
for bar in ax2.patches:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             int(bar.get_height()), ha="center", va="bottom", fontsize=11)

# 3. Hire rate by gender (bar)
ax3 = fig.add_subplot(gs[0, 2])
rate_data = gender_summary.reset_index()
sns.barplot(data=rate_data, x="Gender", y="Hire_Rate", palette=palette, ax=ax3)
ax3.set_title("Hire Rate by Gender (%)")
ax3.set_ylabel("Hire Rate (%)")
ax3.set_ylim(0, 100)
for bar in ax3.patches:
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=11)

# 4. Predicted Fit distribution by gender (stacked bar)
ax4 = fig.add_subplot(gs[1, 0:2])
fit_order = ["very bad", "bad", "average", "good", "very good"]
fit_pct = test.groupby(["Gender", "Predicted_Fit"]).size().unstack(fill_value=0)
for col in fit_order:
    if col not in fit_pct.columns:
        fit_pct[col] = 0
fit_pct = fit_pct[fit_order]
fit_pct_norm = fit_pct.div(fit_pct.sum(axis=1), axis=0) * 100
fit_pct_norm.T.plot(kind="bar", ax=ax4, color=["#4C72B0", "#DD8452"])
ax4.set_title("Predicted Fit Distribution by Gender (%)")
ax4.set_xlabel("Predicted Fit")
ax4.set_ylabel("% of Gender Group")
ax4.set_xticklabels(fit_order, rotation=30)
ax4.legend(title="Gender")

# 5. Hire rate by job and gender
ax5 = fig.add_subplot(gs[1, 2])
job_rate = test.groupby(["Job", "Gender"])["Hired"].mean().reset_index()
job_rate["Hired"] *= 100
sns.barplot(data=job_rate, x="Hired", y="Job", hue="Gender",
            palette=palette, orient="h", ax=ax5)
ax5.set_title("Hire Rate by Job & Gender (%)")
ax5.set_xlabel("Hire Rate (%)")
ax5.set_ylabel("")
ax5.legend(title="Gender", loc="lower right")

plt.savefig("bias_analysis_200.png", dpi=150, bbox_inches="tight")
print("\nBias analysis chart saved → bias_analysis_200.png")
