# 🫀 Cardiovascular Risk Analysis — Healthcare Analytics Case Study

> **For Recruiters:** This project demonstrates how I translate healthcare data into actionable clinical decisions. The focus is on **business-aligned metrics**, **risk-aware decision-making**, and **operational feasibility**—not just model accuracy.

---

## 📊 What This Project Delivers

A **patient risk stratification system** that helps clinical teams identify high-risk individuals for preventive care:

- **77% of disease cases detected** (vs. 0% with a majority-class baseline)
- **Operationally feasible alert volume** (15-20% of patients flagged)
- **67% reduction in missed diagnoses** through optimized decision criteria
- **Designed for real-world deployment** with clear clinical workflows

**Key Insight:** In healthcare, a model that catches 77% of disease cases while generating manageable alerts is far more valuable than one with 91% accuracy that catches nothing.

---

## 🎯 Business Problem

**Stakeholders:** Primary care clinics, cardiology departments, preventive care programs

**Challenge:** How do we identify which patients need follow-up screening when resources are limited?

**Risk Asymmetry:** Missing a sick patient (false negative) is far more costly than ordering an unnecessary test (false positive)

**Success Criteria:** Maximize disease detection while keeping alert volume operationally realistic

---

## 📁 Dataset

- **Source:** [CDC BRFSS 2020](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
- **Size:** 319,795 patient records
- **Features:** 18 health indicators (age, BMI, smoking status, diabetes, etc.)
- **Target:** Heart disease presence (binary)
- **Challenge:** Severe class imbalance (~91% healthy, ~9% disease)

**Why This Matters:** With 91% of patients healthy, a naïve model can achieve 91% accuracy by predicting everyone as healthy—while detecting zero disease cases. This project shows why **accuracy is the wrong metric** for imbalanced healthcare problems.

---

## 🔍 Analytical Approach

### 1. Data Quality Assessment
- Validated 319K records for completeness (no missing values)
- Confirmed appropriate data types and value ranges
- Identified severe class imbalance requiring specialized evaluation metrics

### 2. Exploratory Analysis
- Analyzed distribution of key risk factors (age, smoking, BMI, stroke history)
- Validated that majority-class baseline achieves 91% accuracy with **0% disease detection**
- Established that recall (sensitivity) must be the primary evaluation metric

### 3. Feature Analysis
Identified top risk predictors through comparative analysis of disease rates across groups:
- **Age:** Strong positive association with disease prevalence
- **Smoking Status:** Significant risk factor across all age groups
- **Stroke History:** Strongest single predictor of heart disease
- **BMI & Diabetes:** Secondary risk indicators

### 4. Model Development
- Built logistic regression models with class-balanced weighting
- Evaluated performance using **recall, precision, F1-score, and ROC-AUC**
- Explicitly avoided accuracy as a primary metric due to class imbalance

### 5. Decision Threshold Optimization
- Tuned probability thresholds to balance recall vs. alert volume
- Selected **threshold = 0.3** (vs. default 0.5) to prioritize disease detection
- Quantified tradeoffs: 67% reduction in missed diagnoses at cost of 15-20% alert rate

---

## 📈 Results

### Model Performance (Optimized Threshold)

| Metric | Value | Business Interpretation |
|--------|-------|------------------------|
| **Recall (Sensitivity)** | 77% | Catches 77 out of 100 disease cases |
| **Alert Volume** | 15-20% | Flags 150-200 patients per 1,000 screened |
| **Missed Diagnoses** | ↓ 67% | Compared to default threshold |
| **ROC-AUC** | 0.82 | Strong discriminative ability |

### What This Means in Practice

**Per 1,000 patients screened (estimated based on model performance):**
- **Expected disease cases:** ~90 (9% prevalence)
- **Cases detected by model:** ~69 (77% recall)
- **Missed diagnoses:** ~21 (23% of disease cases)
- **Total patients flagged:** ~200-250 (includes true positives + false positives)
- **Precision:** ~28-35% (roughly 1 in 3 flagged patients has disease)

**Clinical Value:** While only ~1 in 3 alerts represents true disease, this is still a **7-8x improvement** over the 9% base rate. The model concentrates risk effectively while maintaining operationally feasible alert volumes.

---

## 🏥 Deployment Workflow

This isn't just a model—it's a **decision-support system**. Here's how it would work in practice:

```
1. Patient Visit
   ↓
2. Risk Scoring (model assigns probability)
   ↓
3. Automated Flagging (patients above threshold marked as high-risk)
   ↓
4. Clinical Review (physician orders confirmatory tests)
   ↓
5. Outcome Tracking (results feed back into system for recalibration)
```

**Key Design Principle:** The model **supports** clinical decisions—it doesn't replace physician judgment.

---

## 🛠️ Technical Stack

- **Python:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Modeling:** Logistic regression with class weighting
- **Evaluation:** Custom threshold optimization, precision-recall analysis
- **Notebook:** Jupyter for reproducible analysis

---

## 📂 Repository Structure

```
cardiovascular-disease-prediction/
├── heart_disease_case_study.ipynb    # Main analysis notebook
├── data/
│   └── heart_2020_cleaned.csv        # CDC BRFSS dataset
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook heart_disease_case_study.ipynb
```

---

## 💡 Key Skills Demonstrated

- **Business-First Problem Framing:** Translated a healthcare challenge into measurable objectives aligned with clinical workflows
- **Data Quality Assessment:** Validated dataset integrity and identified structural challenges (class imbalance) that drive analytical strategy
- **Decision-Oriented Analytics:** Designed evaluation metrics around operational impact (recall vs. alert volume) rather than statistical abstraction
- **Risk-Aware Interpretation:** Quantified tradeoffs between false positives and false negatives in terms of real-world consequences
- **Stakeholder Communication:** Presented technical findings through the lens of actionable clinical decisions

---

## ⚠️ Limitations & Future Work

### Current Limitations
- Dataset is cross-sectional (no longitudinal tracking)
- Limited to self-reported health indicators
- No integration with EHR systems or lab results
- No real-time monitoring or model drift detection

### Potential Enhancements
- Incorporate temporal features (patient history over time)
- Add lab results (cholesterol, blood pressure readings)
- Implement cost-sensitive learning tied to clinical outcomes
- Build continuous monitoring and recalibration pipeline
- A/B test different threshold strategies in clinical settings

---

## 🎓 What Makes This Different

Most portfolio projects stop at *"I built a model with X% accuracy."*

This project goes further by:
1. **Framing the problem** in terms of stakeholder needs (not just technical metrics)
2. **Choosing metrics** that align with real-world risk (recall over accuracy)
3. **Optimizing decisions** based on operational constraints (alert volume)
4. **Designing deployment** workflows that show systems thinking
5. **Communicating value** in business terms (missed diagnoses, alerts per 1,000 patients)

**The real deliverable isn't the model—it's the decision framework.**

---

## 📧 Questions?

This project is part of my data analytics portfolio. If you'd like to discuss the methodology, results, or potential applications, feel free to reach out!

---

**Built by Abhi Dhindsa** | [GitHub](https://github.com/AbHi23d) | [LinkedIn](https://www.linkedin.com/in/abhi-dhindsa)
