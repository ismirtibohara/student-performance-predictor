# 🎓 Student Performance Predictor

A machine learning project that predicts students' final grades (G3) from study habits, attendance, and past performance.

## 📌 Problem Statement
Early identification of students at risk helps teachers and students act sooner. This project explores the UCI Student Performance dataset and builds a model to predict final grades.

## 🛠 Tech Stack
Python · pandas · SQLite · scikit-learn · Matplotlib/Seaborn · Streamlit

## 📊 Progress
- [x] Week 1: Project setup, data loading, SQL database, data quality check
- [x] Week 2: SQL analysis and charts
- [x] Week 3: Exploratory data analysis
- [x] Week 4: Data cleaning and feature encoding
- [x] Week 5: Model training and comparison
- [ ] Week 6: Model tuning
- [ ] Week 7: Streamlit app
- [ ] Week 8: Deployment and final README

## 📁 Dataset
[UCI Student Performance dataset](https://archive.ics.uci.edu/) (Cortez & Silva, 2008). 395 students, 33 features, no missing values.

## 📈 Results
| Features | Model | MAE | R² |
|---|---|---|---|
| With G1/G2 | Linear Regression | 1.65 | 0.724 |
| With G1/G2 | Random Forest | 1.17 | 0.811 |
| Without G1/G2 | Linear Regression | 3.40 | 0.141 |
| Without G1/G2 | Random Forest | 3.08 | 0.270 |

Earlier grades (G1, G2) are by far the strongest predictors. Without them, predicting final grades is much harder.

## 🚀 Run Locally
```bash
git clone https://github.com/ismirtibohara/student-performance-predictor.git
cd student-performance-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 👩‍💻 Author
Smriti Bohara · BSc CSIT, Tribhuvan University

