# XGBoost example: mean + direct variance modeling (no log transform)
import numpy as np
from xgboost import XGBRegressor

# Toy dataset (same as before)
X = np.array([[1,1],[1,1],[1,1],[2,2],[2,2],[2,2]])
y = np.array([ 2, -2,  3,  2, -3,  4])

# ---- Step 1: Fit mean model ----
xgb_mean = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=0
)
xgb_mean.fit(X, y)
y_pred_mean = xgb_mean.predict(X)

# ---- Step 2: Fit variance model directly on residual² ----
residuals = (y - y_pred_mean)**2
y_var_target = residuals  # no log transform

xgb_var = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=1
)
xgb_var.fit(X, y_var_target)

# ---- Step 3: Predict mean + variance at test inputs ----
X_test = np.array([[1,1],[2,2]])
mean_pred = xgb_mean.predict(X_test)
var_pred = xgb_var.predict(X_test)
std_pred = np.sqrt(var_pred)

# ---- Step 4: Display results ----
for x, mu, sigma in zip(X_test, mean_pred, std_pred):
    print(f"x={x} → predicted mean={mu:.3f}, predicted std={sigma:.3f}")