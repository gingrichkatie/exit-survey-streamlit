
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load cleaned data
df = pd.read_csv("cleaned_exit_survey_data.csv")

# Define dissatisfaction-related columns
reason_cols = [
    'Job dissatisfaction', 'Dissatisfaction with the department',
    'Workload', 'Work life balance', 'Lack of recognition',
    'Opportunities for promotion', 'Physical work environment',
    'Employment conditions', 'Staff morale', 'Workplace issue'
]

# Prepare data
X_raw = df[reason_cols + ['Employment Status', 'Classification', 'Region']]
y = df['dissatisfied']
X = pd.get_dummies(X_raw, drop_first=True)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Sidebar inputs
st.sidebar.header("Survey Inputs")
input_data = {}

for col in reason_cols:
    input_data[col] = 1 if st.sidebar.selectbox(f"{col}?", ["Yes", "No"]) == "Yes" else 0

# Collect categorical variables
employment_status = st.sidebar.selectbox("Employment Status", df['Employment Status'].dropna().unique())
classification = st.sidebar.selectbox("Classification", df['Classification'].dropna().unique())
region = st.sidebar.selectbox("Region", df['Region'].dropna().unique())

# Encode categorical inputs
input_df = pd.DataFrame([input_data])
full_input = pd.concat([input_df,
    pd.get_dummies(pd.DataFrame({
        'Employment Status': [employment_status],
        'Classification': [classification],
        'Region': [region]
    }), drop_first=True)
], axis=1)

# Align with training data
full_input = full_input.reindex(columns=X.columns, fill_value=0)

# Predict
if st.sidebar.button("Predict Dissatisfaction"):
    prediction = model.predict(full_input)[0]
    st.sidebar.success("Prediction: Dissatisfied" if prediction else "Prediction: Not Dissatisfied")

# Main app
st.title("Exit Survey Analysis App")
st.write("This app predicts whether an employee was dissatisfied based on structured exit survey inputs.")

# Charts
st.subheader("Most Common Reasons for Dissatisfaction")
reason_counts = df[reason_cols].sum().sort_values(ascending=False)
st.bar_chart(reason_counts)

st.subheader("Feature Importance (Logistic Coefficients)")
coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values()
st.bar_chart(coefficients.tail(10))

# Confusion Matrix
st.subheader("Model Confusion Matrix")
y_pred = model.predict(X)
conf_matrix = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
