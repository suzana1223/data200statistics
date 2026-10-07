"""Salary Predictor app, built from our Colab notebook.
Run:  streamlit run app.py   (Salary_Data.csv must be in the same folder)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Salary Predictor", page_icon="💰")
FEATURES = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]


@st.cache_data
def load_clean():
    """Same cleaning steps as the notebook."""
    df = pd.read_csv("Salary_Data.csv")
    df = df.drop_duplicates().dropna()
    df = df[df["Salary"] >= 10000]
    df["Education Level"] = df["Education Level"].replace(
        {"Bachelor's Degree": "Bachelor's", "Master's Degree": "Master's", "phD": "PhD"})
    return df


@st.cache_resource
def train(df):
    """Same model as the notebook: get_dummies + Linear Regression."""
    X = pd.get_dummies(df[FEATURES], drop_first=True)
    y = df["Salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"R²": r2_score(y_test, pred),
               "MAE": mean_absolute_error(y_test, pred),
               "RMSE": np.sqrt(mean_squared_error(y_test, pred))}
    return model, list(X.columns), metrics, y_test, pred


df = load_clean()
model, columns, metrics, y_test, y_pred = train(df)

st.title("💰 Employee Salary Predictor")
tab1, tab2, tab3 = st.tabs(["Predict", "Data insights", "Model performance"])

# ---------------------------------------------------------------- Predict
with tab1:
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 18, 70, 30)
    experience = c1.number_input("Years of experience", 0.0, 45.0, 5.0, step=0.5)
    gender = c1.selectbox("Gender", sorted(df["Gender"].unique()))
    education = c2.selectbox("Education level", sorted(df["Education Level"].unique()))
    job = c2.selectbox("Job title", sorted(df["Job Title"].unique()))

    if st.button("Predict salary", type="primary"):
        if experience > age - 16:
            st.warning("Years of experience looks too high for this age.")
        row = pd.DataFrame([{"Age": age, "Gender": gender, "Education Level": education,
                             "Job Title": job, "Years of Experience": experience}])
        # turn text into 0/1 columns, keep only columns the model was trained on
        row = pd.get_dummies(row).reindex(columns=columns, fill_value=0).astype(float)
        st.success(f"Predicted salary: **{max(model.predict(row)[0], 0):,.0f}**")
        st.caption("An estimate from our dataset only (no currency is given in the data).")

# --------------------------------------------------------- Data insights
with tab2:
    st.write(f"Cleaned dataset: **{len(df)} employees**")
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    sns.scatterplot(data=df, x="Years of Experience", y="Salary", ax=ax[0, 0])
    ax[0, 0].set_title("Experience vs Salary")
    sns.boxplot(data=df, x="Education Level", y="Salary", ax=ax[0, 1])
    ax[0, 1].set_title("Salary by Education")
    sns.histplot(df["Salary"], kde=True, ax=ax[1, 0])
    ax[1, 0].set_title("Salary Distribution")
    sns.heatmap(df[["Age", "Years of Experience", "Salary"]].corr(), annot=True, ax=ax[1, 1])
    ax[1, 1].set_title("Correlation")
    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------------------------------- Model performance
with tab3:
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{metrics['R²']:.3f}")
    m2.metric("MAE", f"{metrics['MAE']:,.0f}")
    m3.metric("RMSE", f"{metrics['RMSE']:,.0f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set(xlabel="Actual Salary", ylabel="Predicted Salary", title="Actual vs Predicted")
    st.pyplot(fig)

    st.write("**Effect of each factor** (change in salary, other things equal)")
    coefs = pd.Series(model.coef_, index=columns)
    st.dataframe(coefs[~coefs.index.str.startswith("Job Title")].round(0).rename("Coefficient"))
