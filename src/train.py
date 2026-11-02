from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

ROOT = Path(__file__).resolve().parent.parent

# Load and encode the data
df = pd.read_csv(ROOT / "data" / "student-mat.csv", sep=";")
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop(columns=["G3"])
y = df_encoded["G3"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Tune a Random Forest
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
grid.fit(X_tr, y_tr)

best = grid.best_estimator_
pred = best.predict(X_te)
print("Best settings:", grid.best_params_)
print("MAE:", round(mean_absolute_error(y_te, pred), 2))
print("R2:", round(r2_score(y_te, pred), 3))

# Save the model for the Streamlit app
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(best, ROOT / "models" / "rf_model.joblib")
joblib.dump(list(X.columns), ROOT / "models" / "feature_columns.joblib")
print("Saved model to models/")