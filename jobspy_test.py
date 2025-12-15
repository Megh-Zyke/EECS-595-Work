import csv
from jobspy import scrape_jobs
import pandas as pd

queries = [

    "Sales / Business Development",
    "Marketing / Brand Management",
    "Retail / Store Management",
    "Public Relations / Communications",
    "Event Planning / Coordination",
    "Customer Success / Client Relations",
    "Front Desk / Reception / Office Administration",
    "Travel & Tourism",
    "Real Estate / Property Management",
    "Healthcare / Nursing / Patient Care",
    "Video Production / Editing",
    "Photography / Videography",
    "Animation / Motion Graphics",
    "Social Media Management",
    "Advertising / Campaign Strategy",
    "Fashion / Styling / Merchandising",
    "Music / Sound Design",
    "Art / Illustration",
    "Interior Design / Architecture",
    "Maintenance / Facilities Management",
    "Construction / Skilled Trades",
    "Logistics / Delivery / Transportation",
    "Food & Beverage / Culinary",
    "Beauty / Wellness / Spa Services",
    "Fitness / Personal Training / Sports Coaching",
    "Security / Safety Services",
    "Cleaning / Housekeeping",
    "Childcare / Elderly Care",
    "Landscaping / Gardening"
]


final_jobs = None

for query in queries:
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "zip_recruiter", "google"],# ,"glassdoor", "bayt", "naukri", "bdjobs"],
        search_term= query,
        google_search_term="customer-facing, creative, or practical services jobs since the last one month",
        results_wanted=100,
        hours_old=72*30,
        sleep_time=2,
        )
    
    print(f"Fetched {len(jobs)} jobs for query: {query}")
    if final_jobs is None:
        final_jobs = jobs
    else:
        final_jobs = pd.concat([final_jobs, jobs], ignore_index=True)
jobs = final_jobs.drop_duplicates().reset_index(drop=True)



print(f"Found {len(jobs)} jobs")
print(jobs.shape)
print(jobs.head())

jobs.to_csv("jobs_hospitality.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False) # to_excel