from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "models" / "rf_model.joblib")
    columns = joblib.load(ROOT / "models" / "feature_columns.joblib")
    return model, columns


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "student-mat.csv", sep=";")


model, feature_columns = load_model()
data = load_data()

# Default value for every column the form does not ask about
defaults = {}
for col in data.columns.drop("G3"):
    if pd.api.types.is_numeric_dtype(data[col]):
        defaults[col] = data[col].median()
    else:
        defaults[col] = data[col].mode()[0]

st.title("🎓 Student Performance Predictor")
st.write("Enter a student's details to predict their final grade (G3, out of 20).")

col1, col2 = st.columns(2)
with col1:
    g1 = st.slider("First period grade (G1)", 0, 20, 10)
    g2 = st.slider("Second period grade (G2)", 0, 20, 10)
    studytime = st.selectbox(
        "Weekly study time", [1, 2, 3, 4],
        format_func=lambda x: {1: "<2 hours", 2: "2-5 hours", 3: "5-10 hours", 4: ">10 hours"}[x],
        index=1,
    )
    failures = st.selectbox("Past class failures", [0, 1, 2, 3])
    absences = st.slider("Absences", 0, 75, 4)
with col2:
    age = st.slider("Age", 15, 22, 17)
    sex = st.selectbox("Sex", ["F", "M"])
    medu = st.slider("Mother's education (0-4)", 0, 4, 2)
    fedu = st.slider("Father's education (0-4)", 0, 4, 2)
    higher = st.selectbox("Wants higher education", ["yes", "no"])
    internet = st.selectbox("Internet at home", ["yes", "no"])

if st.button("Predict final grade"):
    inputs = defaults.copy()
    inputs.update({
        "G1": g1, "G2": g2, "studytime": studytime, "failures": failures,
        "absences": absences, "age": age, "sex": sex, "Medu": medu,
        "Fedu": fedu, "higher": higher, "internet": internet,
    })
    row = pd.DataFrame([inputs])
    row = pd.get_dummies(row).reindex(columns=feature_columns, fill_value=0).astype(float)

    prediction = float(model.predict(row)[0])
    prediction = max(0.0, min(20.0, prediction))

    st.metric("Predicted final grade", f"{prediction:.1f} / 20")
    st.caption("Typical error of this model is about 1.2 points when G1 and G2 are known.")

st.divider()
st.caption("Built by Smriti Bohara · Random Forest trained on the UCI Student Performance dataset.")