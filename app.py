# ==============================
# Imports
# ==============================
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from model import train_model, compare_models

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="AI Score Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🎓 AI Student Score Prediction Dashboard")
st.markdown("An interactive machine learning dashboard for predicting student exam scores.")

# ==============================
# Sidebar Settings
# ==============================
st.sidebar.header("⚙️ Model Settings")

model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Linear Regression", "Polynomial Regression"]
)

degree = 1 if model_type == "Linear Regression" else st.sidebar.slider("Select Polynomial Degree", 2, 3, 2)

# ==============================
# Train Model
# ==============================
if "model_data" not in st.session_state:
    st.session_state.model_data = train_model(degree)

if st.sidebar.button("Retrain Model"):
    st.session_state.model_data = train_model(degree)

model, df, r2, mse, poly, X_train, X_test, y_train, y_test, predictions = st.session_state.model_data

# ==============================
# Model Performance
# ==============================
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)
col1.metric("R² Score", round(r2, 3))
col2.metric("MSE", round(mse, 2))

# ==============================
# Prediction Section
# ==============================
st.subheader("🔮 Make Prediction")

hours = st.slider("Select Study Hours", 0, 12, 5)

hours_array = poly.transform([[hours]])
prediction = model.predict(hours_array)[0]

st.success(f"Predicted Score: {round(prediction,2)}")

# Download CSV
prediction_df = pd.DataFrame({
    "Hours": [hours],
    "Predicted Score": [round(prediction, 2)],
    "Model Type": [model_type],
    "Degree": [degree]
})

st.download_button(
    label="Download Prediction as CSV",
    data=prediction_df.to_csv(index=False),
    file_name="prediction_result.csv",
    mime="text/csv"
)

# ==============================
# Regression Plot
# ==============================
st.subheader("📈 Train vs Test with Regression Line")

fig = go.Figure()

# Train Data
fig.add_scatter(
    x=X_train[:,0],
    y=y_train,
    mode="markers",
    name="Train Data"
)

# Test Data
fig.add_scatter(
    x=X_test[:,0],
    y=y_test,
    mode="markers",
    name="Test Data"
)

# Regression Line
x_range = np.linspace(0, 12, 100)
x_poly = poly.transform(x_range.reshape(-1,1))
y_range = model.predict(x_poly)

fig.add_scatter(
    x=x_range,
    y=y_range,
    mode="lines",
    name="Regression Line"
)

# Highlight user prediction
fig.add_scatter(
    x=[hours],
    y=[prediction],
    mode="markers",
    marker=dict(size=12),
    name="Your Prediction"
)

fig.update_layout(
    xaxis_title="Hours",
    yaxis_title="Score"
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# Model Equation (Linear Only)
# ==============================
if degree == 1:
    coef = model.coef_[1]
    intercept = model.intercept_
    st.subheader("📐 Model Equation")
    st.write(f"Score = {round(coef,2)} * Hours + {round(intercept,2)}")

# ==============================
# Residual Plot
# ==============================
st.subheader("📉 Residual Plot")

residuals = y_test - predictions

fig_res = go.Figure()

fig_res.add_scatter(
    x=X_test[:,0],
    y=residuals,
    mode="markers",
    name="Residuals"
)

fig_res.add_hline(y=0)

fig_res.update_layout(
    xaxis_title="Hours",
    yaxis_title="Residual"
)

st.plotly_chart(fig_res, use_container_width=True)

# ==============================
# Model Comparison
# ==============================
st.subheader("🏆 Model Performance Comparison (R² Score)")

comparison_results = compare_models()

comparison_df = pd.DataFrame({
    "Model": list(comparison_results.keys()),
    "R2 Score": list(comparison_results.values())
})

fig_compare = px.bar(
    comparison_df,
    x="Model",
    y="R2 Score",
    title="Comparison of Model Performance"
)

st.plotly_chart(fig_compare, use_container_width=True)

# ==============================
# Explanation Section
# ==============================
st.subheader("📘 Model Explanation")

st.markdown("""
### Linear Regression
Predicts score using a straight line:

Score = aX + b

### Polynomial Regression
Adds higher-degree terms:

Score = a₂X² + a₁X + b

### R² Score
Measures how well the model explains variance.

### MSE
Measures average squared prediction error.
""")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Scikit-Learn")
st.markdown("Author: Winyou | AI Student Project")