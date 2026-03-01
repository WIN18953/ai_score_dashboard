import streamlit as st
import plotly.express as px
import numpy as np
from model import train_model

st.title("AI Student Score Prediction Dashboard")

degree = st.slider("Select Polynomial Degree", 1, 3, 1)

model, df, r2, mse, poly, X_test, y_test, predictions = train_model(degree)

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

# กราฟ
fig = px.scatter(df, x="Hours", y="Score", title="Hours vs Score")

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