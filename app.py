import streamlit as st
import plotly.express as px
import numpy as np
from model import train_model

st.title("AI Student Score Prediction Dashboard")

model, df, r2, mse, coef, intercept = train_model()

# แสดงค่าโมเดล
st.subheader("Model Performance")

col1, col2 = st.columns(2)
col1.metric("R² Score", round(r2, 3))
col2.metric("MSE", round(mse, 2))

# Slider ให้เลือกชั่วโมงอ่าน
hours = st.slider("Select Study Hours", 0, 12, 5)

# ทำนาย
prediction = model.predict([[hours]])[0]

st.subheader("Prediction Result")
st.success(f"Predicted Score: {round(prediction,2)}")

# กราฟ
fig = px.scatter(df, x="Hours", y="Score", title="Hours vs Score")

x_range = np.linspace(0, 12, 100)
y_range = model.predict(x_range.reshape(-1,1))

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