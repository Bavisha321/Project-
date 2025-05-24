# model.py
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Load and train the model
def train_model():
    df = pd.read_csv("synthetic_cibil_scores.csv")
    X = df.drop("CIBIL_Score", axis=1)
    y = df["CIBIL_Score"]

    model = xgb.XGBRegressor()
    model.fit(X, y)

    # Evaluate model
    #y_pred = model.predict(X)
    #rmse = mean_squared_error(y, y_pred, squared=False)
    #r2 = r2_score(y, y_pred)

    #print(f"Model RMSE: {rmse:.2f}")
    #print(f"Model R2 Score: {r2:.2f}")
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    print(f"Model RMSE: {rmse:.2f}")
    print(f"Model R2 Score: {r2:.2f}")


    # Create SHAP explainer
    explainer = shap.Explainer(model)
    return model, explainer

# Predict and explain
def predict_cibil(model, explainer, values):
    cols = ["Payment_History", "Credit_Utilization", "Credit_Age",
            "Number_of_Accounts", "Hard_Inquiries", "Debt_to_Income_Ratio"]
    df_input = pd.DataFrame([dict(zip(cols, values))])
    score = model.predict(df_input)[0]
    shap_values = explainer(df_input)
    suggestions = get_suggestions(df_input, shap_values)
    return score, shap_values, suggestions

# Suggest improvements for negative SHAP values
def get_suggestions(df_input, shap_values):
    feature_names = df_input.columns.tolist()
    shap_val = shap_values.values[0]
    suggestions = []

    for i, val in enumerate(shap_val):
        if val < 0:
            feature = feature_names[i]
            tip = get_tip(feature, df_input.iloc[0][feature])
            suggestions.append(f"🔻 **{feature}** is negatively affecting your score. Tip: {tip}")

    return suggestions

def get_tip(feature, value):
    tips = {
        "Payment_History": "Aim for a consistent payment history close to 100%.",
        "Credit_Utilization": "Try to keep credit utilization below 30%.",
        "Credit_Age": "Avoid closing old accounts to improve average credit age.",
        "Number_of_Accounts": "Avoid opening too many new accounts quickly.",
        "Hard_Inquiries": "Limit the number of credit inquiries over a short period.",
        "Debt_to_Income_Ratio": "Reduce debt or increase income to lower this ratio."
    }
    return tips.get(feature, "General financial discipline can help.")
