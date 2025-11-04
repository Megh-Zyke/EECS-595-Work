from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

import re

def preprocess_gender_decoder_output(raw_text: str):
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    
    # 1. Detect bias type (masculine-coded / feminine-coded / neutral)
    bias_match = re.search(r"(masculine|feminine|neutral)-coded", raw_text, re.IGNORECASE)
    bias_type = bias_match.group(0).lower() if bias_match else "unknown"

    # 2. Extract masculine-coded words
    masculine_words = []
    feminine_words = []

    # Find indices for the masculine and feminine sections
    try:
        masc_start = lines.index("Masculine-coded words in this ad") + 1
        masc_end = lines.index("See the full list of masculine-coded words")
        masculine_words = lines[masc_start:masc_end]
    except ValueError:
        masculine_words = []

    try:
        fem_start = lines.index("Feminine-coded words in this ad") + 1
        fem_end = lines.index("See the full list of feminine-coded words")
        # Handle the "No feminine-coded words were found." case
        if "No feminine-coded words were found." in lines[fem_start:fem_end]:
            feminine_words = []
        else:
            feminine_words = lines[fem_start:fem_end]
    except ValueError:
        feminine_words = []

    return {
        "bias_type": bias_type,
        "masculine_words": masculine_words,
        "feminine_words": feminine_words,
    }


def analyze_gender_bias(job_description: str):
    url = "https://gender-decoder.katmatfield.com/"

    # Set up headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    # Find the textarea and enter text
    textarea = driver.find_element(By.ID, "texttotest")
    textarea.clear()
    textarea.send_keys(job_description)

    # Find and click the submit button
    submit_button = driver.find_element(By.XPATH, "//input[@type='submit']")
    submit_button.click()

    # Wait for JavaScript to update the page
    time.sleep(2)

    # Extract the results
    try:
        result_div = driver.find_element(By.ID, "results")
        print("=== Gender Decoder Result ===")
        output = preprocess_gender_decoder_output(result_div.text)
        print(output)
    except:
        print("Could not find the result section — site structure may have changed.")

    driver.quit()

# Example usage
job_text = """We are looking for an ambitious strong and driven individual to join our dynamic team of engineers from asia."""
analyze_gender_bias(job_text)
