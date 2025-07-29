
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Streamlit settings
st.set_page_config(page_title="Exit Survey Analysis", layout="wide")
st.title("📊 Exit Survey Dissatisfaction Classifier")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dete_survey.csv")  # <-- Replace with your file name if needed
    return df

df = load_data()
st.markdown("### Raw Survey Preview")
st.dataframe(df.head())

# Rename dissatisfaction-related columns for clarity
renamed_columns = {
    "Job dissatisfaction": "Unhappy with job tasks",
    "Dissatisfaction with the department": "Unhappy with department leadership",
    "Physical work environment": "Unsatisfactory physical work environment",
    "Lack of recognition": "Felt unrecognized",
    "Lack of job security": "Concerned about job security",
    "Work location": "Unhappy with work location",
    "Employment conditions": "Poor employment conditions",
    "Interpersonal conflicts": "Interpersonal conflict at work"
}
df = df.rename(columns=renamed_columns)

# Define dissatisfaction factors
reason_cols = list(renamed_columns.values())
optional_cols = ['Employment Status', 'Classification', 'Region']

# Select only columns that exist in the file
existing_cols = [col for col in reason_cols + optional_cols if col in df.columns]
survey_df = df[existing_cols].copy()

# Convert Yes/No to 1/0
survey_df = survey_df.replace({'Yes': 1, 'No': 0}).fillna(0)

# Create target column
survey_df["Overall Dissatisfaction"] = survey_df[reason_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

# Show processed preview
st.markdown("### Cleaned Survey Preview")
st.dataframe(survey_df.head())

# Define features and target
X = survey_df[reason_cols]
y = survey_df["Overall Dissatisfaction"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification report
st.subheader("Model Performance")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Satisfied", "Dissatisfied"], yticklabels=["Satisfied", "Dissatisfied"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig_cm)

# Feature Importance Plot
st.subheader("Top Factors Contributing to Dissatisfaction")
feature_importance = pd.Series(model.coef_[0], index=X.columns)
fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax_fi)
ax_fi.set_title("Feature Importance (Logistic Regression Coefficients)")
st.pyplot(fig_fi)

# Save output file (optional)
survey_df.to_csv("final_cleaned_survey.csv", index=False)
