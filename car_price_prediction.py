"""
CodeAlpha - Data Science Internship
Task 3: Car Price Prediction with Machine Learning

Predicts the selling price of used cars from features such as
present price, year, kms driven, fuel type, seller type, transmission
and number of previous owners.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
OUT_DIR = "outputs"
CURRENT_YEAR = 2024  # dataset predates 2024; used to compute car age

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("car_data.csv")
print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# ---------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------
# Car age is far more predictive than the raw manufacture year
df["Car_Age"] = CURRENT_YEAR - df["Year"]
df = df.drop(columns=["Year", "Car_Name"])  # name is high-cardinality/unique, drop it

# ---------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(df["Selling_Price"], kde=True, color="steelblue")
plt.title("Distribution of Selling Price (lakhs)")
plt.savefig(f"{OUT_DIR}/price_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="Present_Price", y="Selling_Price", hue="Fuel_Type")
plt.title("Present Price vs Selling Price")
plt.savefig(f"{OUT_DIR}/present_vs_selling_price.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 4. Preprocessing: one-hot encode categoricals
# ---------------------------------------------------------
df_encoded = pd.get_dummies(
    df, columns=["Fuel_Type", "Selling_type", "Transmission"], drop_first=True
)

X = df_encoded.drop(columns=["Selling_Price"])
y = df_encoded["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 5. Train & compare models
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"\n=== {name} ===")
    print(f"MAE:  {mae:.3f} lakhs")
    print(f"RMSE: {rmse:.3f} lakhs")
    print(f"R2:   {r2:.4f}")

# ---------------------------------------------------------
# 6. Best model + feature importance + actual vs predicted
# ---------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_name]
print(f"\nBest model: {best_name} (R2={results[best_name]['R2']:.4f})")

preds = best_model.predict(X_test_scaled)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds, alpha=0.7, color="darkorange")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title(f"Actual vs Predicted - {best_name}")
plt.savefig(f"{OUT_DIR}/actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="seagreen")
    plt.title(f"Feature Importance - {best_name}")
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

# Model comparison chart
plt.figure(figsize=(7, 4))
plt.bar(results.keys(), [v["R2"] for v in results.values()], color="teal")
plt.ylabel("R2 Score")
plt.title("Model Comparison (R2)")
plt.xticks(rotation=20, ha="right")
plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nAll plots saved to outputs/")
