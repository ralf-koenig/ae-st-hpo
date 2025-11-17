# XGBoost example: mean + variance modeling for repeated noisy targets
import numpy as np
from xgboost import XGBRegressor

# Toy dataset with repeated inputs and varying outputs
X = np.array([[1,1],[1,1],[1,1],[2,2],[2,2],[2,2]])
y = np.array([ 2, -2,  3,  2, -3,  4])

# ---- Step 1: Fit XGBoost to predict the mean ----
xgb_mean = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=0
)
xgb_mean.fit(X, y)

# Predict mean at each training point
y_pred_mean = xgb_mean.predict(X)

# ---- Step 2: Fit a second XGBoost to predict log-variance ----
residuals = (y - y_pred_mean) ** 2
# Use log(residual^2 + small epsilon) to stabilize
y_var_target = np.log(residuals + 1e-6)

xgb_var = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=1
)
xgb_var.fit(X, y_var_target)

# Predict mean and variance for new points
X_test = np.array([[1,1],[2,2]])
mean_pred = xgb_mean.predict(X_test)
log_var_pred = xgb_var.predict(X_test)
var_pred = np.exp(log_var_pred)
std_pred = np.sqrt(var_pred)

# ---- Step 3: Print results ----
for x, mu, sigma in zip(X_test, mean_pred, std_pred):
    print(f"x={x} → predicted mean={mu:.3f}, predicted std={sigma:.3f}")
