import streamlit as st
import plotly.express as px
import numpy as np
from model import train_model

st.sidebar.header("Model Settings")

model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Linear Regression", "Polynomial Regression"]
)

if model_type == "Linear Regression":
    degree = 1
else:
    degree = st.sidebar.slider("Select Polynomial Degree", 2, 3, 2)

if "model_data" not in st.session_state:
    st.session_state.model_data = train_model(degree)

if st.sidebar.button("Retrain Model"):
    st.session_state.model_data = train_model(degree)

model, df, r2, mse, poly, X_train, X_test, y_train, y_test, predictions = st.session_state.model_data

# แสดงค่าโมเดล
st.subheader("Model Performance")

col1, col2 = st.columns(2)
col1.metric("R² Score", round(r2, 3))
col2.metric("MSE", round(mse, 2))

# Slider ให้เลือกชั่วโมงอ่าน
hours = st.slider("Select Study Hours", 0, 12, 5)

# ทำนาย
hours_array = poly.transform([[hours]])
prediction = model.predict(hours_array)[0]

st.subheader("Prediction Result")
st.success(f"Predicted Score: {round(prediction,2)}")

import pandas as pd

# Create prediction dataframe
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

# กราฟ
import plotly.graph_objects as go

fig = go.Figure()

# Train points
fig.add_scatter(
    x=poly.inverse_transform(X_train)[:,1],
    y=y_train,
    mode="markers",
    name="Train Data"
)

# Test points
fig.add_scatter(
    x=poly.inverse_transform(X_test)[:,1],
    y=y_test,
    mode="markers",
    name="Test Data"
)

# Regression line
x_range = np.linspace(0, 12, 100)
x_poly = poly.transform(x_range.reshape(-1,1))
y_range = model.predict(x_poly)

fig.add_scatter(
    x=x_range,
    y=y_range,
    mode="lines",
    name="Regression Line"
)

fig.update_layout(
    title="Train vs Test with Regression Line",
    xaxis_title="Hours",
    yaxis_title="Score"
)

st.plotly_chart(fig)

x_range = np.linspace(0, 12, 100)
x_poly = poly.transform(x_range.reshape(-1,1))
y_range = model.predict(x_poly)

fig.add_scatter(x=x_range, y=y_range, mode="lines", name="Regression Line")

fig.add_scatter(
    x=[hours],
    y=[prediction],
    mode="markers",
    marker=dict(size=12),
    name="Your Prediction"
)

st.plotly_chart(fig)
st.subheader("Model Equation")
st.write(f"Score = {round(coef,2)} * Hours + {round(intercept,2)}")

# Residual Plot
st.subheader("Residual Plot")

residuals = y_test - predictions

import plotly.graph_objects as go

fig_res = go.Figure()

fig_res.add_scatter(
    x=X_test["Hours"],
    y=residuals,
    mode="markers",
    name="Residuals"
)

fig_res.add_hline(y=0)

fig_res.update_layout(
    xaxis_title="Hours",
    yaxis_title="Residual",
    title="Residual vs Hours"
)

st.plotly_chart(fig_res)