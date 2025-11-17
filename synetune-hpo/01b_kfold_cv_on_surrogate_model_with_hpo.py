from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.blackbox_surrogate import add_surrogate

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

if __name__ == "__main__":
    for ae_architecture in ['vanillix', 'varix']:
        for (datascenario, task) in [
            ('tcga', 'rna'),
            ('tcga', 'meth'),
            ('tcga', 'dna'),
            ('schc', 'rna'),
            ('schc', 'meth')
        ]:

            blackbox = load_blackbox(f"autoencodix_{ae_architecture}")[f"{datascenario}_{task}"]

            Xr = blackbox.hyperparameters  # shape (3000, 7/8)
            yr = pd.DataFrame(
                blackbox.objectives_evaluations.reshape(3000, 3),
                columns=blackbox.objectives_names
            )  # shape (3000, 3)

            df_reg = Xr.copy()
            df_reg["target"] = yr["WEIGHTED_AVG_AUC_DOWNSTREAM_PERFORMANCE"]

            n_splits = 5
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            rmse_per_fold = []
            r2_per_fold = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(df_reg.drop(columns="target")), start=1):
                X_train_full = df_reg.iloc[train_idx].drop(columns="target").values
                y_train_full = df_reg.iloc[train_idx]["target"].values

                X_test = df_reg.iloc[test_idx].drop(columns="target").values
                y_test = df_reg.iloc[test_idx]["target"].values

                # Split training into training/validation for hyperparameter tuning
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_full, y_train_full, test_size=0.2, random_state=42
                )

                # Define parameter grid for tuning
                param_grid = { # 4 x 3 x 3 x 3 x 3 = 324 options
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 300],
                    'subsample': [0.7, 0.8, 1.0],
                    'colsample_bytree': [0.7, 0.8, 1.0]
                }

                best_params = None
                best_rmse = np.inf

                # Randomized search over parameter grid
                for i in range(50):  # try 10 random combinations
                    params = {k: np.random.choice(v) for k, v in param_grid.items()}
                    model = xgb.XGBRegressor(**params,
                                             random_state=42,
                                             n_jobs=-1,
                                             early_stopping_rounds=20,
                                             verbosity=0)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                    preds_val = model.predict(X_val)
                    rmse_val = root_mean_squared_error(y_val, preds_val)

                    if rmse_val < best_rmse:
                        best_rmse = rmse_val
                        best_params = params

                # Train best model on the full training data (train + val)
                best_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
                best_model.fit(X_train_full, y_train_full)

                predictions = best_model.predict(X_test)

                plt.scatter(y_test, predictions, marker='+', label=f"Fold {fold}", alpha=0.7)

                rmse = root_mean_squared_error(y_test, predictions)
                rmse_per_fold.append(rmse)

                r2 = r2_score(y_test, predictions)
                r2_per_fold.append(r2)

                print(f" Fold {fold} best params: {best_params} | val RMSE={best_rmse:.4f}")

            plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
            plt.xlabel("Real AUC_OVO")
            plt.ylabel("Predicted AUC_OVO")
            plt.title(f"{ae_architecture}: {datascenario}_{task}: Mean R2: {np.mean(r2_per_fold):.4f} ± {np.std(r2_per_fold):.4f}")
            plt.xlim([.4, 1])
            plt.ylim([.4, 1])
            plt.legend()
            plt.show()

            print(f'RUN: {ae_architecture} - {datascenario} - {task}\n',
                  f"Mean R2: {np.mean(r2_per_fold):.4f} ± {np.std(r2_per_fold):.4f}",
                  f"Mean RMSE: {np.mean(rmse_per_fold):.4f} ± {np.std(rmse_per_fold):.4f}")
