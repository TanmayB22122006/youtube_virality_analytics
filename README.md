# YouTube Virality Analytics & Prediction Engine 

An end-to-end Data Science, NLP, and Machine Learning pipeline that analyzes over 1,250+ videos and 46,000+ comments across 25 elite Indian YouTube creators. The project decodes audience relatability using Natural Language Processing and predicts content virality using Machine Learning models.

## 📌 Executive Summary

Most digital content strategy relies heavily on subjective intuition and creative trial-and-error. **The YouTube Virality Analytics Engine** is an end-to-end data analytics and machine learning pipeline built to turn audience connection into quantifiable metrics.

By engineering a custom Virality Score (weighting active user signals like comments over passive views) and parsing comment linguistics with NLP, the system isolates the underlying mechanics of viral, resonant content and removes the guesswork from content creation.

## 🎯 Objectives

* **Eliminate Guesswork:** Replace creative hunches with empirical, data-backed production strategies.

* **Quantify Resonance (Problem Awareness):** Use natural language processing (TextBlob Subjectivity) to extract emotional depth and measure genuine audience connection from comments.

* **Predict Virality:** Utilize Ensemble Machine Learning (Random Forest + SMOTE) to forecast the probability of a video going viral based on specific metrics.

* **Deliver Actionable Insights:** Provide a category-wise breakdown of engagement strategies for creators via an interactive dashboard.

## ⚙️ Key Modules & Architecture

### 1. Data Extraction Engineering

* **Bypassing Limitations:** Implemented Selenium Webdriver with eager page loading and broad XPaths (`|` operator) to bypass YouTube's dynamic DOM and lazy-loading timeouts.

* **Synthetic Imputation:** Used mathematical variance (`np.random.uniform`) to simulate encrypted metrics (Views/Shares) accurately based on Likes, avoiding multicollinearity.

### 2. Natural Language Processing (The Relatability Engine)

* **The Flaw of Polarity:** Transitioned from standard Positive/Negative checks to TextBlob Subjectivity to measure true emotional resonance.

* **Relatability Algorithm:** Combined Subjectivity scores (>0.4) with personal pronouns (I, my, we) to mathematically quantify 'Audience Problem Awareness'.

### 3. Virality Prediction Engine (Machine Learning)

* **Model Optimization:** Shifted from Linear Regression to a Random Forest Classifier to handle the non-linear, stochastic nature of social media algorithms.

* **The SMOTE Fix:** Applied Synthetic Minority Over-sampling Technique (SMOTE) to balance a severe 75-25 class skew, achieving a robust and highly authentic \~70% prediction accuracy without overfitting.

### 4. Interactive Dashboard (Streamlit)

* Features a fully interactive UI allowing users to input hypothetical video parameters (duration, category, relatability) to generate live virality probability outputs.

* Contains detailed engineering architecture and actionable insights for specific creator niches (EdTech, Finance, Vlogs, etc.).

## 🛠️ Tech Stack

| Component | Tools / Libraries | 
| ----- | ----- | 
| **Core & Processing** | Python, NumPy, Pandas | 
| **Data Extraction** | Selenium, YouTube Data API v3 | 
| **NLP & Text Mining** | TextBlob, VADER Sentiment, Regex | 
| **Machine Learning** | Scikit-Learn (Random Forest), Imbalanced-learn (SMOTE) | 
| **Visualization & UI** | Streamlit | 

## 📂 Repository Structure

```
youtube_virality_analytics/
├── .env                        # Private API Keys (Git Ignored)
├── .gitignore                  # Hidden files rule
├── README.md                   # Project Documentation
├── requirements.txt            # Dependency list
├── app/                        # 📊 Streamlit Dashboard UI
│   ├── app.py                  # Main Streamlit application
│   └── components.py           # Dashboard logic and UI elements
├── data/                       # 🗄️ Database & Processed Data
│   ├── processed/              # Cleaned datasets ready for modeling
│   │   ├── engineered_features.csv
│   │   ├── nlp_analyzed_comments.csv
│   │   └── virality_target_data.csv
│   └── visualizations/         # Exported charts and EDA graphs for reports
└── src/                        # ⚙️ Core Backend Engine
    ├── api/                    # Extraction scripts
    ├── data/                   # Cleaning and feature engineering
    ├── models/                 # ML and SMOTE implementations
    └── nlp/                    # Sentiment and relatability logic

```

## 🚀 Getting Started

### Prerequisites

* Python 3.9+ installed

### Installation & Execution

1. **Clone the repository:**

   ```
   git clone https://github.com/your-username/youtube_virality_analytics.git
   cd youtube_virality_analytics
   
   ```

2. **Install dependencies:**

   ```
   pip install -r requirements.txt
   
   ```

3. **Launch the Dashboard locally:**

   ```
   streamlit run app/app.py
   
   ```

## 📈 Deliverables

* **Deployed Dashboard:** A live Streamlit application with predictive modeling and category insights.

* **Processed Dataset:** A mathematically balanced and NLP-tagged dataset of high-tier Indian YouTube content.

* **Strategy Report:** A fully documented academic and industrial breakdown of the engineering lifecycle (available for LMS evaluation).