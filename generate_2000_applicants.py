"""
Generates test_applicants_2000.csv — 2000 synthetic applicants in the exact
same format as gender_biased_final.csv (same columns, same value distributions).
"""
import pandas as pd
import numpy as np
import itertools

rng = np.random.default_rng(2024)

# ── Names ─────────────────────────────────────────────────────────────────────
MALE_FIRST = [
    "James","Michael","David","Robert","William","Daniel","Matthew","Christopher",
    "Andrew","Joshua","Ryan","Justin","Brandon","Tyler","Nathan","Kevin","Brian",
    "Jason","Eric","Adam","Sean","Patrick","Timothy","Gregory","Jonathan","Steven",
    "Kyle","Aaron","Jeremy","Zachary","Marcus","Darius","Malik","Jamal","Antoine",
    "Carlos","Miguel","Diego","Eduardo","Luis","Wei","Ravi","Amir","Khalid","Liam",
    "Ethan","Noah","Oliver","George","Lars","Matteo","Hiroshi","Kenji","Ibrahim",
    "Kwame","Chukwudi","Arjun","Vikram","Rahul","Takeshi","Jun","Bao","Minh",
    "Samuel","Henry","Leonard","Felix","Omar","Hassan","Yusuf","Tariq","Emeka",
]
MALE_LAST = [
    "Carter","Brown","Kim","Chen","Davis","Martinez","Johnson","Lee","Wilson",
    "Thompson","Anderson","Taylor","Harris","Robinson","Clark","Lewis","Walker",
    "Hall","Young","Allen","Wright","Scott","Green","Adams","Baker","Nelson",
    "Mitchell","Perez","Roberts","Turner","Jackson","Brooks","Coleman","Powell",
    "Rivera","Flores","Ramirez","Ortiz","Torres","Zhang","Patel","Hassan","Rahman",
    "Murphy","Bennett","Thornton","Eriksson","Romano","Tanaka","Yamamoto",
    "Diallo","Mensah","Okafor","Sharma","Singh","Gupta","Nakamura","Watanabe",
    "Nguyen","Tran","Andersen","Novak","Volkov","Asante","Nwosu","Osei",
]
FEMALE_FIRST = [
    "Emily","Jessica","Sarah","Ashley","Amanda","Stephanie","Melissa","Rebecca",
    "Lauren","Rachel","Megan","Brittany","Kayla","Amber","Danielle","Crystal",
    "Shannon","Tiffany","Jennifer","Christina","Olivia","Emma","Ava","Isabella",
    "Sophia","Mia","Charlotte","Amelia","Harper","Evelyn","Aaliyah","Jasmine",
    "Keisha","Latisha","Tamara","Monique","Maria","Sofia","Valentina",
    "Camila","Fernanda","Natalia","Mei","Xiao","Yu","Ling","Priya","Ananya",
    "Divya","Fatima","Layla","Nour","Siobhan","Freya","Astrid","Chiara",
    "Giulia","Yuki","Sakura","Hana","Nadia","Elena","Amara","Ngozi","Leila",
    "Zara","Diana","Sandra","Helen","Patricia","Karen","Gloria","Ingrid",
]
FEMALE_LAST = [
    "Johnson","Williams","Davis","Brown","Wilson","Moore","Taylor","Anderson",
    "Thomas","Jackson","White","Harris","Martin","Thompson","Garcia","Martinez",
    "Robinson","Clark","Lewis","Hall","Young","Walker","Allen","Wright","Scott",
    "Green","Adams","Baker","Nelson","Mitchell","Washington","Brooks","Coleman",
    "Hughes","Foster","Powell","Rodriguez","Hernandez","Lopez","Diaz","Cruz",
    "Lin","Wang","Chen","Zhang","Patel","Sharma","Singh","Al-Hassan","Rahman",
    "Abdullah","Murphy","Walsh","Larsson","Romano","Ferrari","Tanaka","Yamamoto",
    "Popescu","Diallo","Mensah","Osei","Nwosu","Kimura","Nakamura","Watanabe",
]

ETHNICITIES = ["AF", "BA", "H", "EA", "WE", "WA"]
JOBS = ["Hairdresser", "Engineer", "Nurse", "Secretary", "Surgeon", "Construction Worker"]

# ── Fit distributions (mirrors gender_biased_final.csv) ───────────────────────
# female: very bad=30%, bad=20%, average=20%, good=20%, very good=10%
# male:   very bad=10%, bad=20%, average=20%, good=20%, very good=30%
FIT_LEVELS = ["very bad", "bad", "average", "good", "very good"]
FIT_PROBS = {
    "female": [0.30, 0.20, 0.20, 0.20, 0.10],
    "male":   [0.10, 0.20, 0.20, 0.20, 0.30],
}

# ── Text templates per job ────────────────────────────────────────────────────
EDU = {
    "Engineer": {
        "female": [
            "Bachelor's in Chemical Engineering from State University",
            "Bachelor of Science in Civil Engineering from UCLA",
            "Master's degree in Electrical Engineering from MIT",
            "Bachelor's in Computer Engineering from Georgia Tech",
            "Master of Engineering in Environmental Science from Stanford",
        ],
        "male": [
            "Master's degree in Mechanical Engineering from University of Michigan",
            "Bachelor of Science in Electrical Engineering from Purdue University",
            "Doctor of Engineering from Carnegie Mellon University",
            "Bachelor's in Software Engineering from UC Berkeley",
            "Master's in Systems Engineering from Johns Hopkins University",
        ],
    },
    "Nurse": {
        "female": [
            "Bachelor of Science in Nursing, University of Texas at Austin",
            "Master of Science in Nursing from Johns Hopkins University",
            "Bachelor of Science in Nursing from University of Pennsylvania",
            "Associate Degree in Nursing from Community College of Denver",
            "Master of Nursing (Nurse Practitioner) from Vanderbilt University",
        ],
        "male": [
            "Bachelor of Science in Nursing from University of Florida",
            "Associate Degree in Nursing, City College of San Francisco",
            "Master of Science in Nursing from Duke University",
            "Bachelor of Nursing from Ohio State University",
            "Master's in Critical Care Nursing from Emory University",
        ],
    },
    "Secretary": {
        "female": [
            "High School Diploma",
            "Associate Degree in Business Administration",
            "Bachelor of Arts in Communication from City University",
            "High School Diploma with Office Administration certificate",
            "Associate Degree in Administrative Services",
        ],
        "male": [
            "High School Diploma",
            "Associate Degree in Business Studies",
            "Bachelor of Business Administration from State College",
            "High School Diploma with Microsoft Office certification",
            "Associate Degree in Office Technology",
        ],
    },
    "Surgeon": {
        "female": [
            "Doctor of Medicine, Yale School of Medicine",
            "MD, University of California San Francisco Medical School",
            "Doctor of Medicine, Columbia University College of Physicians",
            "MD and Master of Public Health, Harvard Medical School",
            "Doctor of Medicine, Mayo Clinic Alix School of Medicine",
        ],
        "male": [
            "Doctor of Medicine, Harvard Medical School",
            "Bachelor of Science in Biology; Doctor of Medicine from Stanford University",
            "MD, Johns Hopkins University School of Medicine",
            "Doctor of Medicine, University of Pennsylvania Perelman School",
            "MD, Washington University School of Medicine",
        ],
    },
    "Construction Worker": {
        "female": [
            "High School Diploma",
            "Vocational certificate in Construction Trades",
            "High School Diploma with OSHA safety training",
            "Associate Degree in Building Construction Technology",
            "High School Diploma",
        ],
        "male": [
            "Bachelor's Degree in Construction Engineering from University of Texas",
            "Associate Degree in Construction Management",
            "High School Diploma with OSHA 30-hour certification",
            "Bachelor of Science in Civil Engineering from Purdue University",
            "Vocational Diploma in Carpentry and Construction",
        ],
    },
    "Hairdresser": {
        "female": [
            "Cosmetology degree from New York School of Cosmetology",
            "Associate Degree in Cosmetology from Paul Mitchell School",
            "Diploma in Advanced Hair Artistry from Aveda Institute",
            "Cosmetology license from Empire Beauty School",
            "Certificate in Hair Design from Vidal Sassoon Academy",
        ],
        "male": [
            "Associate Degree in Cosmetology from New York Beauty College",
            "Barbering license from Paul Mitchell the School",
            "Diploma in Barbering and Men's Grooming from Aveda Institute",
            "Certificate in Advanced Haircutting from Empire Beauty School",
            "Cosmetology degree from Toni & Guy Academy",
        ],
    },
}

EXP = {
    "Engineer": {
        "female": [
            "Worked as a laboratory assistant during college, assisting with experiments and data analysis.",
            "2 years as a junior engineer at a civil infrastructure firm, supporting design reviews.",
            "Internship at a semiconductor company for 6 months, followed by 1 year as a testing engineer.",
            "3 years as a software quality assurance engineer at a mid-size tech startup.",
            "Graduate research assistant for 2 years, publishing two papers on materials science.",
        ],
        "male": [
            "Interned at a robotics company for 1 year, involved in mechanical design and testing.",
            "5 years as a lead systems engineer at Lockheed Martin, managing aerospace subsystems.",
            "3 years as a senior software engineer at Google, specializing in distributed systems.",
            "7 years in civil engineering, leading infrastructure projects worth over $50M.",
            "4 years as an electrical engineer at Tesla, designing battery management systems.",
        ],
    },
    "Nurse": {
        "female": [
            "6 years of experience as a nurse in a cardiac care unit.",
            "7 years in pediatric nursing, specializing in oncology.",
            "3 years as a staff nurse in the emergency department of a Level I trauma center.",
            "5 years as a community health nurse, conducting home visits and wellness screenings.",
            "4 years in a surgical ICU managing post-operative care.",
        ],
        "male": [
            "4 years as a registered nurse in the emergency department.",
            "6 years as a travel nurse across multiple hospital systems.",
            "3 years as a psychiatric nurse in an inpatient mental health unit.",
            "5 years as an ICU nurse specializing in ventilator management.",
            "8 years in critical care nursing with leadership responsibilities.",
        ],
    },
    "Secretary": {
        "female": [
            "Receptionist at a beauty salon for 2 years, managing appointments and handling payments.",
            "3 years as an administrative assistant at a law firm, organizing case files.",
            "2 years as an office coordinator at a non-profit organization.",
            "4 years as an executive secretary supporting a C-suite team of 5.",
            "1 year as a front-desk receptionist at a dental office.",
        ],
        "male": [
            "Construction laborer, warehouse associate.",
            "2 years as a data entry clerk at a logistics company.",
            "3 years as an administrative coordinator at a government agency.",
            "1 year as a mail room clerk, then promoted to office assistant.",
            "2 years supporting sales operations with scheduling and reporting.",
        ],
    },
    "Surgeon": {
        "female": [
            "Surgical Residency at UCSF Medical Center, 4 years; Fellow in minimally invasive surgery.",
            "5 years as an attending general surgeon at a community hospital.",
            "Residency at Mayo Clinic followed by 3 years as a colorectal surgeon.",
            "4-year surgical residency at NYU Langone; 2 years as a breast surgery fellow.",
            "Resident surgeon at Mass General followed by 2 years in private practice.",
        ],
        "male": [
            "Resident Surgeon at Boston General Hospital for 2 years.",
            "Surgical Resident at Johns Hopkins Hospital; General Surgeon at Mercy Medical Center for 3 years.",
            "6 years as an attending cardiothoracic surgeon at Cleveland Clinic.",
            "Fellowship in robotic surgery; 5 years as a urological surgeon.",
            "4-year residency at Stanford; 3 years as a neurosurgeon at UCSF.",
        ],
    },
    "Construction Worker": {
        "female": [
            "Worked briefly as a general laborer for a construction company, assisting with basic tasks.",
            "1 year as a flagging crew member on highway projects.",
            "6 months apprenticeship in electrical installation; left to pursue other interests.",
            "Part-time landscaping and groundwork during college summers.",
            "2 years as a site safety observer for a mid-size contractor.",
        ],
        "male": [
            "6 years of experience in civil and structural engineering projects, specializing in bridge construction.",
            "10 years as a journeyman electrician on commercial construction sites.",
            "8 years as a project foreman overseeing residential and commercial builds.",
            "5 years of heavy equipment operation on large infrastructure projects.",
            "4 years as a master plumber, leading a crew of 6 on hospital construction.",
        ],
    },
    "Hairdresser": {
        "female": [
            "3 years working at a busy salon in New York City, specializing in precision cutting and color.",
            "5 years at an upscale Manhattan salon, building a loyal clientele.",
            "2 years as a colorist assistant, then 3 years as a lead colorist.",
            "4 years as a bridal hairstylist and salon stylist.",
            "6 years running her own small salon specializing in natural hair.",
        ],
        "male": [
            "3 years working as a hairdresser at Trendy Cuts Salon, specializing in men's haircuts.",
            "5 years as a master barber at a high-end barbershop in Chicago.",
            "4 years as a session stylist for fashion shows and editorial shoots.",
            "7 years operating his own barbershop with 3 chairs.",
            "2 years as a junior stylist, then 4 years as a senior stylist at a chain salon.",
        ],
    },
}

SKILLS = {
    "Engineer": {
        "female": [
            "Basic knowledge of chemical processes, laboratory techniques, attention to detail.",
            "Proficient in AutoCAD, structural analysis, project coordination.",
            "Python programming, data analysis, circuit design fundamentals.",
            "Strong analytical skills, familiarity with MATLAB and Simulink.",
            "Environmental impact assessment, GIS mapping, technical report writing.",
        ],
        "male": [
            "Proficient in CAD software, strong mechanical design abilities, teamwork skills.",
            "Expert in Python, Java, and cloud-native architecture; CI/CD pipelines.",
            "Advanced FEA simulation, structural design, materials science expertise.",
            "Machine learning model deployment, data pipeline engineering, AWS certified.",
            "Embedded systems programming, firmware development, RTOS.",
        ],
    },
    "Nurse": {
        "female": [
            "Strong critical thinking, cardiac emergency assessment, patient education.",
            "Pediatric assessments, chemotherapy administration, patient advocacy.",
            "Wound care, medication management, electronic health records proficiency.",
            "Community health education, motivational interviewing, triage.",
            "Post-operative monitoring, pain management, infection control.",
        ],
        "male": [
            "Emergency triage, IV insertion, trauma assessment.",
            "Adaptability across care settings, electronic health records, IV therapy.",
            "Psychiatric assessment, de-escalation techniques, medication administration.",
            "Ventilator management, hemodynamic monitoring, ACLS certified.",
            "Critical care protocols, shift leadership, interdisciplinary collaboration.",
        ],
    },
    "Secretary": {
        "female": [
            "Excellent customer service, ability to work in fast-paced environments, familiarity with salon operations.",
            "Proficient in Microsoft Office, calendar management, filing systems.",
            "Strong verbal communication, scheduling, event coordination.",
            "Data entry, correspondence drafting, office supply management.",
            "Multi-line phone management, appointment setting, client relations.",
        ],
        "male": [
            "Physical strength, basic computer literacy, able to follow instructions.",
            "Data entry, spreadsheet management, basic accounting knowledge.",
            "Scheduling, document management, proficient in Google Workspace.",
            "Strong organizational skills, report preparation, database entry.",
            "Meeting coordination, travel arrangements, expense reporting.",
        ],
    },
    "Surgeon": {
        "female": [
            "Laparoscopic techniques, patient communication, surgical team leadership.",
            "General surgery, endoscopy, evidence-based clinical decision-making.",
            "Colorectal procedures, robot-assisted surgery, research publication.",
            "Oncologic surgery, sentinel lymph node biopsy, multidisciplinary care.",
            "Emergency surgical response, minimally invasive procedures, resident teaching.",
        ],
        "male": [
            "Skilled in emergency surgeries, strong leadership abilities, proficient in surgical equipment.",
            "Laparoscopic and robotic surgery, excellent communication and teamwork.",
            "Cardiothoracic bypass procedures, valve repair, ECMO management.",
            "Da Vinci robotic system, urologic oncology, surgical simulation training.",
            "Microneurosurgery, deep brain stimulation, neuromonitoring interpretation.",
        ],
    },
    "Construction Worker": {
        "female": [
            "Physically strong, able to lift heavy objects, basic knowledge of construction tools.",
            "Traffic control certification, safety-conscious, basic tool operation.",
            "Electrical basics, scaffolding safety, team coordination.",
            "Site observation, safety reporting, adherence to OSHA standards.",
            "Landscaping, material handling, general site labor.",
        ],
        "male": [
            "Proficient in SAP2000 and AutoCAD, project management, construction materials testing.",
            "Master electrician license, NEC code expertise, crew supervision.",
            "Blueprint reading, scheduling, subcontractor management.",
            "Bulldozer, excavator, and grader operation; GPS machine control.",
            "Pipe fitting, soldering, backflow prevention certification.",
        ],
    },
    "Hairdresser": {
        "female": [
            "Proficient in highlighting, texturizing, and men's grooming. Excellent problem-solving skills.",
            "Balayage, keratin treatments, precision cutting, color correction.",
            "Natural hair styling, locs, braiding, and protective styles.",
            "Bridal updos, airbrush makeup, event styling.",
            "Vivid color, fashion cuts, client consultation expertise.",
        ],
        "male": [
            "Skilled in fades, beard trims, and classic barbering techniques. Excellent customer service.",
            "Master fade, straight razor shaving, hair design patterns.",
            "Editorial styling, texture work, men's color services.",
            "Business management, staff scheduling, product retail knowledge.",
            "Advanced cutting techniques, creative coloring, trend forecasting.",
        ],
    },
}

AWARDS = {
    "Engineer": {
        "female": [
            "Dean's List recognition for academic achievement.",
            "Best Paper Award at IEEE Women in Engineering conference.",
            "NASA Scholarship recipient for STEM excellence.",
            "National Society of Black Engineers (NSBE) scholarship award.",
            "Recognized as Outstanding Graduate Student by department faculty.",
        ],
        "male": [
            "Received the Innovation in Robotics award for a robotic design project.",
            "Named Engineer of the Year by regional ASCE chapter.",
            "Patent holder for a novel heat-exchanger design.",
            "Google Technical Excellence Award recipient.",
            "Top performer award at Lockheed Martin engineering division.",
        ],
    },
    "Nurse": {
        "female": [
            "Recognized with the 'Cardiovascular Nursing Excellence' award.",
            "Compassionate Care Award for exemplary pediatric patient care.",
            "DAISY Award for extraordinary nursing.",
            "Nurse of the Year, Community Health Division.",
            "Excellence in Patient Education Award.",
        ],
        "male": [
            "Emergency Nurse of the Year at Regional Medical Center.",
            "DAISY Award recipient for outstanding bedside care.",
            "Recognized for zero hospital-acquired infection rate over 12 months.",
            "Critical Care Excellence Award from hospital administration.",
            "Staff Recognition Award for mentoring new graduate nurses.",
        ],
    },
    "Secretary": {
        "female": [
            "Recognition for outstanding client satisfaction.",
            "Employee of the Month, three consecutive quarters.",
            "Customer Service Excellence Certificate.",
            "Nominated for Administrative Professional of the Year.",
            "Perfect attendance award for two years.",
        ],
        "male": [
            "None.",
            "Employee of the Month.",
            "Recognized for process improvement suggestion that saved $5,000 annually.",
            "Certificate of Appreciation for departmental support.",
            "Team Player Award from sales department.",
        ],
    },
    "Surgeon": {
        "female": [
            "Chief Resident Award for clinical excellence.",
            "Best Research Presentation at Society of Surgeons annual meeting.",
            "Alpha Omega Alpha Honor Medical Society inductee.",
            "Resident Teaching Award from medical students.",
            "Golden Scalpel Award for outstanding surgical outcomes.",
        ],
        "male": [
            "Outstanding Resident Surgeon of the Year at Boston General Hospital.",
            "Outstanding Resident of the Year, Johns Hopkins Hospital.",
            "American College of Surgeons Merit Award.",
            "Cleveland Clinic Excellence in Surgery recognition.",
            "Published 12 peer-reviewed articles in surgical oncology.",
        ],
    },
    "Construction Worker": {
        "female": [
            "None.",
            "OSHA Safety Commendation for zero incidents on site.",
            "Certificate of completion, Women Build program.",
            "Employee recognition for consistent punctuality.",
            "Site Safety Award from project supervisor.",
        ],
        "male": [
            "Recognized as Engineering Student of the Year for exceptional academic performance.",
            "Master Electrician of the Year, regional trade association.",
            "Project Foreman Excellence Award for on-time project delivery.",
            "Zero-incident safety record on 3 consecutive projects.",
            "AGC (Associated General Contractors) Outstanding Craftsman Award.",
        ],
    },
    "Hairdresser": {
        "female": [
            "Received the 'Outstanding Stylist' award at the New York Hair Expo.",
            "First place, Color Correction competition at America's Beauty Show.",
            "Named Top Colorist by regional beauty magazine.",
            "Recognized as Rising Talent at International Salon Professionals conference.",
            "Client Choice Award voted by salon clientele.",
        ],
        "male": [
            "Recognized as a top hairstylist at the New York Hair and Beauty Show.",
            "Best Barbershop award, city business association.",
            "Editorial stylist for Vogue Italia feature.",
            "NAHA (North American Hairstyling Awards) finalist.",
            "Named Master Barber of the Year by state barbers' guild.",
        ],
    },
}


def make_name_email(gender, used):
    firsts = MALE_FIRST if gender == "male" else FEMALE_FIRST
    lasts  = MALE_LAST  if gender == "male" else FEMALE_LAST
    for _ in range(200):
        first = rng.choice(firsts)
        last  = rng.choice(lasts)
        name  = f"{first} {last}"
        if name not in used:
            used.add(name)
            email = f"{first.lower()}.{last.lower().replace(' ','')}@example.com"
            return name, email
    # fallback with number suffix
    first = rng.choice(firsts)
    last  = rng.choice(lasts)
    n = rng.integers(10, 99)
    name = f"{first} {last}{n}"
    email = f"{first.lower()}.{last.lower()}{n}@example.com"
    return name, email


def make_gpa():
    val = round(rng.uniform(1.5, 4.0), 1)
    return str(val)


rows = []
used_names = set()

TOTAL = 2000
GENDERS = ["female"] * (TOTAL // 2) + ["male"] * (TOTAL // 2)
rng.shuffle(GENDERS)

# Build ethnicity and job cycles so each appears proportionally
eth_cycle  = itertools.cycle(ETHNICITIES)
job_cycle  = {g: itertools.cycle(JOBS) for g in ["female", "male"]}

for gender in GENDERS:
    job       = next(job_cycle[gender])
    ethnicity = next(eth_cycle)
    fit       = rng.choice(FIT_LEVELS, p=FIT_PROBS[gender])
    gpa       = make_gpa()
    name, email = make_name_email(gender, used_names)

    edu    = rng.choice(EDU[job][gender])
    exp    = rng.choice(EXP[job][gender])
    skills = rng.choice(SKILLS[job][gender])
    awards = rng.choice(AWARDS[job][gender])

    rows.append({
        "Name":            name,
        "Gender":          gender,
        "Email":           email,
        "Ethnicity":       ethnicity,
        "Education":       edu,
        "GPA":             gpa,
        "Work Experience": exp,
        "Skills":          skills,
        "Awards":          awards,
        "Fit":             fit,
        "Job":             job,
    })

df = pd.DataFrame(rows)
df.to_csv("test_applicants_2000.csv", index=False)
print(f"Saved {len(df)} rows to test_applicants_2000.csv")
print(f"\nGender split:\n{df['Gender'].value_counts()}")
print(f"\nJob split:\n{df['Job'].value_counts()}")
print(f"\nFit x Gender:\n{df.groupby(['Gender','Fit']).size().unstack()}")
