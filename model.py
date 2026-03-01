import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def train_model():
    df = pd.read_csv("data.csv")

    X = df[["Hours"]]
    y = df["Score"]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)

    return model, df, r2, mse