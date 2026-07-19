"""
One-time script — generates test_applicants.csv (200 fictional individuals).
Run this first, then run run_model_on_applicants.py to get predictions.
"""
import pandas as pd
import numpy as np

np.random.seed(7)

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

JOBS = ["Hairdresser", "Engineer", "Nurse", "Secretary", "Surgeon", "Construction Worker"]
ETHNICITIES = ["AF", "BA", "H", "EA", "WE", "WA"]

CITIES = [
    "New York", "Chicago", "Houston", "Phoenix", "Los Angeles",
    "Seattle", "Boston", "Atlanta", "Denver", "Miami",
]
SALONS   = ["Trendy Cuts Salon", "Elite Style Studio", "The Hair Lounge", "Urban Chic Salon", "Prestige Hair Co"]
HOSPITALS = ["City General Hospital", "Metro Medical Center", "St. Luke's Hospital", "University Hospital", "Riverside Clinic"]
COMPANIES = ["TechCorp", "BuildRight Inc", "Apex Solutions", "Metro Group", "Nexus Industries", "Summit Engineering"]

EDUCATION = {
    "Hairdresser": [
        "Associate Degree in Cosmetology from {city} Beauty College",
        "Bachelor's in Cosmetology from {city} School of Cosmetology",
        "Certificate in Hairdressing from {city} Institute of Beauty",
        "Diploma in Barbering and Styling from {city} Vocational School",
    ],
    "Engineer": [
        "Bachelor's in Mechanical Engineering from {city} State University",
        "Master's in Electrical Engineering from {city} Tech Institute",
        "Bachelor's in Civil Engineering from University of {city}",
        "PhD in Computer Engineering from {city} Institute of Technology",
        "Bachelor's in Software Engineering from {city} Polytechnic",
    ],
    "Nurse": [
        "Bachelor's in Nursing from {city} University Medical School",
        "Associate Degree in Nursing from {city} Community College",
        "Master's in Nursing Practice from {city} Health Sciences University",
        "Bachelor of Science in Nursing from {city} General Hospital School of Nursing",
    ],
    "Secretary": [
        "Associate Degree in Business Administration from {city} Community College",
        "Bachelor's in Office Management from {city} State College",
        "Certificate in Administrative Support from {city} Business Institute",
        "Bachelor's in Communications from {city} University",
    ],
    "Surgeon": [
        "MD from {city} School of Medicine",
        "PhD in Surgical Sciences from {city} Medical University",
        "Master's in Surgery from {city} General Hospital Medical School",
        "MD and residency training at {city} Academic Medical Center",
    ],
    "Construction Worker": [
        "Certificate in Construction Technology from {city} Vocational Institute",
        "Associate Degree in Building Trades from {city} Technical College",
        "Bachelor's in Construction Management from {city} State University",
        "Apprenticeship Certificate in Carpentry from {city} Trades Council",
    ],
}

WORK_EXP = {
    "Hairdresser": [
        "{name} has {n} years of experience at {salon}, specializing in precision cuts and advanced color techniques.",
        "{name} worked for {n} years as a lead stylist at {salon}, recognized for client satisfaction and creativity.",
        "{name} has {n} years of barbering and styling experience at {salon}, excelling in modern and classic techniques.",
    ],
    "Engineer": [
        "{name} has {n} years of engineering experience at {company}, leading cross-functional technical projects.",
        "{name} spent {n} years at {company} designing mechanical systems and managing product development cycles.",
        "{name} has {n} years at {company} developing scalable software solutions and mentoring junior engineers.",
    ],
    "Nurse": [
        "{name} has {n} years of clinical nursing experience at {hospital}, providing compassionate patient-centered care.",
        "{name} worked for {n} years in the ICU at {hospital}, specializing in critical care and emergency response.",
        "{name} has {n} years of experience in pediatric nursing at {hospital}, known for excellent family communication.",
    ],
    "Secretary": [
        "{name} has {n} years of administrative experience supporting C-suite executives at {company}.",
        "{name} managed all office operations for {n} years at {company}, coordinating calendars, travel, and communications.",
        "{name} has {n} years of experience as an executive assistant at {company}, handling confidential correspondence.",
    ],
    "Surgeon": [
        "{name} has {n} years of surgical experience at {hospital}, performing complex minimally invasive procedures.",
        "{name} completed {n} years of residency and surgical practice at {hospital}, specializing in orthopedic surgery.",
        "{name} has {n} years of experience as a cardiac surgeon at {hospital}, with an outstanding patient outcome record.",
    ],
    "Construction Worker": [
        "{name} has {n} years of hands-on construction experience at {company}, skilled in structural and finishing work.",
        "{name} worked on major infrastructure projects for {n} years with {company}, operating heavy equipment safely.",
        "{name} has {n} years of experience in residential and commercial construction at {company}.",
    ],
}

SKILLS = {
    "Hairdresser": [
        "Proficient in cutting, coloring, highlighting, and styling. Strong customer service and interpersonal skills.",
        "Skilled in balayage, keratin treatments, and precision cuts. Excellent time management in fast-paced salon environments.",
    ],
    "Engineer": [
        "Skilled in CAD, project management, and systems design. Proficient in Python, MATLAB, and AutoCAD.",
        "Expert in machine learning, data pipelines, and cloud infrastructure. Strong analytical and problem-solving skills.",
    ],
    "Nurse": [
        "Expert in patient assessment, IV therapy, and medication management. Strong empathy and crisis communication.",
        "Skilled in wound care, patient education, and electronic health records. Calm under pressure with excellent bedside manner.",
    ],
    "Secretary": [
        "Proficient in MS Office Suite, scheduling software, and correspondence drafting. Excellent multitasking and discretion.",
        "Skilled in calendar management, travel coordination, and meeting facilitation. Fast typing with superior attention to detail.",
    ],
    "Surgeon": [
        "Expert in laparoscopic and robotic surgical techniques. Strong decision-making under pressure and exceptional dexterity.",
        "Skilled in trauma surgery and post-operative care. Proficient in surgical simulation and resident teaching.",
    ],
    "Construction Worker": [
        "Skilled in carpentry, concrete work, scaffolding, and equipment operation. Strong safety awareness and physical endurance.",
        "Proficient in blueprint reading, site coordination, and OSHA compliance. Reliable team player with strong work ethic.",
    ],
}

AWARDS = {
    "Hairdresser": [
        "{name} received the 'Best Stylist' award at the {city} Hair and Beauty Expo.",
        "{name} was named Top Hairdresser of the Year by {salon}.",
    ],
    "Engineer": [
        "{name} received the 'Innovation Award' from the {city} Engineering Society.",
        "{name} was recognized as Engineer of the Quarter at {company} for outstanding project delivery.",
    ],
    "Nurse": [
        "{name} received the 'Excellence in Patient Care' award from {hospital}.",
        "{name} was named Nurse of the Year at {hospital} for exceptional dedication.",
    ],
    "Secretary": [
        "{name} was awarded 'Administrative Professional of the Year' at {company}.",
        "{name} received the 'Outstanding Support Staff' recognition at the {city} Business Excellence Gala.",
    ],
    "Surgeon": [
        "{name} received the 'Outstanding Surgeon' award from the {city} Medical Association.",
        "{name} was recognized for surgical excellence and innovation at {hospital}'s annual awards ceremony.",
    ],
    "Construction Worker": [
        "{name} received the 'Safety Excellence' award from {company} for zero-incident project leadership.",
        "{name} was recognized as Top Performer on the {city} Downtown Infrastructure Project.",
    ],
}


def pick(lst):
    return lst[np.random.randint(len(lst))]


def make_person(gender):
    job  = pick(JOBS)
    city = pick(CITIES)
    eth  = pick(ETHNICITIES)
    gpa  = round(np.random.uniform(2.4, 4.0), 1)
    yrs  = np.random.randint(1, 16)

    if gender == "male":
        first = pick(MALE_FIRST)
        last  = pick(MALE_LAST)
    else:
        first = pick(FEMALE_FIRST)
        last  = pick(FEMALE_LAST)

    name  = f"{first} {last}"
    email = f"{first.lower()}.{last.lower().replace(chr(39),'')}@email.com"

    edu   = pick(EDUCATION[job]).format(city=city)
    exp   = pick(WORK_EXP[job]).format(name=first, n=yrs, salon=pick(SALONS),
                                        hospital=pick(HOSPITALS), company=pick(COMPANIES))
    skill = pick(SKILLS[job])
    award = pick(AWARDS[job]).format(name=first, city=city,
                                      salon=pick(SALONS), hospital=pick(HOSPITALS),
                                      company=pick(COMPANIES))

    return {
        "Name":           name,
        "Gender":         gender,
        "Email":          email,
        "Ethnicity":      eth,
        "Education":      edu,
        "GPA":            str(gpa),
        "Work Experience": exp,
        "Skills":         skill,
        "Awards":         award,
        "Fit":            "",
        "Job":            job,
    }


rows = [make_person("male") for _ in range(100)] + \
       [make_person("female") for _ in range(100)]

df = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("test_applicants.csv", index=False)
print(f"Saved test_applicants.csv  ({len(df)} rows)")
print(df["Job"].value_counts().to_string())
