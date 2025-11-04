# Job Description Bias Analysis - Methodology

## Overview
This analysis examined 2,251 job descriptions for various types of bias using pattern matching and keyword detection.

## Bias Categories Analyzed

### 1. **Age Bias (34.7% of jobs)**
Detected language that suggests preferences for younger or older candidates:
- Keywords: "digital native," "young," "energetic," "recent graduate," "early career"
- Experience requirements: "10+ years experience" (can exclude younger workers)
- Terms like "tech-savvy," "dynamic," "millennial," "Gen Z"

### 2. **Gender Bias (41.3% of jobs)**
Identified gendered language and stereotypes:
- Gendered pronouns: "he," "his," "she," "her"
- Gendered job titles: "salesman," "chairman," "foreman," "waitress," "hostess"
- Masculine-coded words: "rockstar," "ninja," "guru," "dominant," "aggressive," "competitive"
- Feminine-coded words: "nurturing," "empathetic," "supportive"
- Informal gendered terms: "guys," "bros," "brotherhood"

### 3. **LGBTQ+ Bias (36.3% of jobs)**
Language suggesting heteronormative or conservative preferences:
- "Traditional values," "family values," "conservative"
- "Christian values," "lifestyle"
- Binary gender language: "he/she"

### 4. **Family Status Bias (33.4% of jobs)**
Assumptions about family commitments:
- "Single," "married," "children," "family-oriented"
- "Work-life balance," "flexible schedule"
- "Extensive travel," "long hours," "weekends required"
- "Willing to relocate"

### 5. **Race/Ethnic Bias (13.8% of jobs)**
Language that may suggest racial preferences:
- "Native English speaker," "American sounding name"
- "Cultural fit," "traditional," "foreign," "non-native"
- "Articulate," "well-spoken" (when not job-relevant)
- "Professional appearance," "grooming standards"
- "Urban," "inner-city" (when used as coded language)

### 6. **Disability Bias (9.6% of jobs)**
Unnecessary physical or mental requirements:
- "Physically fit," "able-bodied," "perfect vision"
- "Must be able to stand/walk/lift/carry"
- "Healthy," "physical stamina," "mental stamina"
- "Must drive," "must have own car"

### 7. **Socioeconomic Bias (7.6% of jobs)**
Preferences for privileged backgrounds:
- "Prestigious university," "top-tier university," "Ivy League"
- "Elite university," "private school"
- "Upscale," "high-end," "luxury," "affluent," "sophisticated"
- "Country club," "golf club"

### 8. **Appearance Bias (2.0% of jobs)**
Unnecessary appearance requirements:
- "Professional appearance," "well-groomed," "presentable"
- "Image conscious," "neat appearance"
- "Dress code," "grooming standards"

### 9. **Proximity Bias (2.1% of jobs)**
Preferences for local candidates:
- "Local candidates only," "must live in/within"
- "No relocation," "must be local"
- "Within X miles," "community member"

### 10. **Affinity Bias (1.0% of jobs)**
Preferences for similar backgrounds:
- "Cultural fit," "team fit," "like-minded"
- "Share our values," "fit in," "one of us"
- "Similar background," "same wavelength"

### 11. **Education Bias (1.7% of jobs)**
Degree requirements that may not be necessary:
- "Must have BA/BS/MA/MS/MBA/PhD"
- "Degree required," "college degree required"
- "Four-year degree," "graduate degree"

## Results Summary

- **Total Jobs Analyzed**: 2,251
- **Jobs with Bias**: 1,933 (85.9%)
- **Jobs without Bias**: 318 (14.1%)

### Most Common Bias Types:
1. Gender (41.3%)
2. LGBTQ+ (36.3%)
3. Age (34.7%)
4. Family Status (33.4%)
5. Race/Ethnicity (13.8%)

## Output Format

The annotated CSV file contains two new columns:

1. **biases_detected**: Pipe-separated list of bias types found (e.g., "gender|age|family_status")
2. **flagged_words**: Pipe-separated list of specific words/phrases that triggered the bias detection

## Important Notes

- Multiple bias types can be detected in a single job description
- The detection is based on keyword/pattern matching and may include false positives
- Some flagged words may be legitimate job requirements depending on context
- This analysis identifies potentially problematic language but requires human review for context
- Words like "lifestyle," "flexible schedule," or "competitive" may be innocuous in some contexts

## Recommendations for Use

1. Review flagged descriptions manually to confirm bias
2. Consider the job context - some physical requirements are legitimate (e.g., firefighter)
3. Use this as a screening tool to identify descriptions that need revision
4. Focus on the most frequent bias types first (gender, LGBTQ+, age)
5. Replace gendered language with neutral alternatives
6. Remove unnecessary requirements that exclude protected groups

## Limitations

- Pattern matching cannot understand context perfectly
- Some legitimate job requirements may be flagged
- Subtle biases not captured by keyword detection may exist
- Analysis is limited to English language patterns
