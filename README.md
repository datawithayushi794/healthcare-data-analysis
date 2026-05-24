# Healthcare Data Analytics — Diabetes Risk Analysis

![Python](https://img.shields.io/badge/Python-3.10-purple?logo=python)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20Analytics-red)
![ML](https://img.shields.io/badge/ML-Logistic_Regression-blue)
![AUC](https://img.shields.io/badge/ROC--AUC-0.673-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Clinical data analysis identifying diabetes risk factors** — combining domain knowledge from an MSc in Biotechnology with data analytics skills. Covers EDA, statistical analysis, visualization, and logistic regression classification.

---

## Project Overview

This project analyzes clinical data from 768 patients to identify the strongest predictors of diabetes. As an MSc Biotechnology graduate, I applied domain expertise in the data cleaning phase — recognizing that zero values in glucose, BMI, and blood pressure are biologically impossible and must be treated as recording errors, not valid zeros.

**Questions answered:**
- Which clinical features most strongly predict diabetes?
- How do glucose and BMI levels differ between diabetic and healthy patients?
- Does diabetes risk increase with age?
- Can logistic regression effectively classify diabetes risk from routine clinical measures?

---

## Key Findings

- **Glucose is the strongest predictor** (correlation r = 0.286 with outcome)
- **Diabetic patients** avg glucose 117.4 mg/dL vs 100.5 mg/dL in healthy patients
- **Obese patients** (BMI > 30) have 71.6% diabetes rate vs 57.8% for normal BMI
- **Age risk**: Diabetes rate rises from ~62% (21-30 age group) to ~67% (60+ age group)
- **Model**: Logistic regression achieved 66% accuracy, ROC-AUC 0.673 on test set
- **Biologically-informed imputation**: 49.5% of insulin values were 0 (impossible) — replaced with outcome-group medians

---

## Dataset

| Property | Details |
|----------|---------|
| Source   | Pima Indians Diabetes Database (Kaggle) |
| Size     | 768 patients × 9 columns |
| Target   | Outcome (0 = No Diabetes, 1 = Diabetes) |
| Features | Glucose, BMI, Age, Pregnancies, BloodPressure, Insulin, SkinThickness, DiabetesPedigreeFunction |

> **Domain Note**: This dataset has a known limitation — zero values appear in Glucose, BloodPressure, BMI, Insulin, and SkinThickness. These are biologically impossible in living patients and are recording errors. I replaced them with outcome-group medians rather than global medians, preserving biological signal between diabetic and non-diabetic populations.

---

## Project Structure

```
healthcare-data-analysis/
├── data/
│   ├── diabetes.csv           → raw dataset
│   └── diabetes_clean.csv     → cleaned with imputation + new features
├── scripts/
│   └── healthcare_analysis.py → cleaning + EDA + charts + ML model
├── outputs/
│   └── figures/               → 6 chart images
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Visualizations

| # | Chart | Description |
|---|-------|-------------|
| 1 | Bar + histogram | Outcome distribution + glucose by status |
| 2 | Box plots (2×3) | 6 clinical features by diabetes status |
| 3 | Correlation heatmap | All feature correlations including outcome |
| 4 | Bar charts | Diabetes rate by Age Group and BMI Category |
| 5 | Scatter plot | Glucose vs BMI colored by outcome |
| 6 | Confusion matrix + ROC curve | Model evaluation |

---

## Machine Learning

```
Algorithm  : Logistic Regression (scikit-learn)
Features   : Glucose, BMI, Age, Pregnancies, BloodPressure, Insulin, DPF
Scaling    : StandardScaler (mean=0, std=1)
Split      : 80% train / 20% test (stratified)
Accuracy   : 66%
ROC-AUC    : 0.673
```

**Feature importance (by coefficient magnitude):**
1. Glucose (0.663)
2. BMI (0.416)
3. Pregnancies (0.256)
4. Insulin (-0.206)

---

## How to Run

```bash
git clone https://github.com/datawithayushi794/healthcare-data-analysis.git
cd healthcare-data-analysis
pip install -r requirements.txt
python scripts/healthcare_analysis.py

-#ayushi srivastava** | MSc Biotechnology | Aspiring Data Analyst  
[www.linkedin.com/in/ayushi-srivastava-6863a9317](#) | [GitHub](#)

