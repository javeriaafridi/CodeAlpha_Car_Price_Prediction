# CodeAlpha_Car_Price_Prediction

**CodeAlpha Data Science Internship — Task 3**

## Objective
Predict the selling price of a used car based on its features (present price, age, mileage, fuel type, seller type, transmission, ownership history).

## Dataset
`car_data.csv` — 301 used car listings with columns:
`Car_Name, Year, Selling_Price, Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner`

## Approach
1. **Feature engineering** — converted `Year` into `Car_Age` (more predictive than raw year); dropped `Car_Name` (high-cardinality, not generalizable).
2. **EDA** — price distribution, correlation heatmap, present vs. selling price scatter by fuel type.
3. **Preprocessing** — one-hot encoded categorical columns (`Fuel_Type`, `Selling_type`, `Transmission`), standardized numeric features.
4. **Modeling** — trained and compared 4 regressors:
   - Linear Regression
   - Lasso Regression
   - Decision Tree
   - Random Forest
5. **Evaluation** — MAE, RMSE, R².

## Results
| Model | MAE (lakhs) | RMSE (lakhs) | R² |
|---|---|---|---|
| Linear Regression | 1.216 | 1.866 | 0.849 |
| Lasso Regression | 1.227 | 1.893 | 0.844 |
| Decision Tree | 0.733 | 1.121 | 0.946 |
| **Random Forest** | **0.624** | **0.954** | **0.961** |

**Random Forest** was the best performer. Feature importance shows `Present_Price` and `Car_Age` are the strongest predictors of selling price.

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python car_price_prediction.py
```
Plots are saved to `outputs/`.

## Files
- `car_price_prediction.py` — full pipeline (feature engineering, EDA, preprocessing, training, evaluation)
- `car_data.csv` — dataset
- `outputs/` — generated plots (price distribution, correlation heatmap, actual vs predicted, feature importance, model comparison)

---
*Submitted as part of the CodeAlpha Data Science Internship.*
