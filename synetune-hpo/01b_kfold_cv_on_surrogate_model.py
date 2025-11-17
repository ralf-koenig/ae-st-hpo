from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.blackbox_surrogate import add_surrogate

import matplotlib.pyplot as plt

# k-fold cross-validation on true AutoEncodix results with 1000 rows in each scenario.

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, r2_score

if __name__ == "__main__":
    for ae_architecture in ['vanillix', 'varix']:
        for (datascenario, task) in [('tcga','rna'),('tcga','meth'),('tcga','dna'),('schc','rna'),('schc','meth')]:

            blackbox = load_blackbox(f"autoencodix_{ae_architecture}")[f"{datascenario}_{task}"]

            Xr = blackbox.hyperparameters #shape (3000, 7/8)
            yr = pd.DataFrame(blackbox.objectives_evaluations.reshape(3000,3),
                              columns= blackbox.objectives_names) # shape (3000, 3)

            df_reg = Xr
            df_reg["target"] = yr["WEIGHTED_AVG_AUC_DOWNSTREAM_PERFORMANCE"]

            # print("\nRegression DataFrame shape:", df_reg.shape)  # should be (3000, 8/9)

            # Modify df inplace to apply filtering before evaluating r²
            # filtered_df = df_reg[df_reg['LATENT_DIM_FIXED'] == 16]
            # filtered_df = df_reg[df_reg['K_FILTER'] == 4096]
            # filtered_df = df_reg[df_reg['BATCH_SIZE'] == 32]
            # df_reg = filtered_df.reset_index(drop=True)

            n_splits = 5
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            # print("\nKFold fold sizes (train/test rows) for regression:")
            # for fold, (train_idx, test_idx) in enumerate(kf.split(df_reg.drop(columns="target")), start=1):
            #    print(f" Fold {fold}: train {len(train_idx)} rows, test {len(test_idx)} rows")

            rmse_per_fold = []
            r2_per_fold = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(df_reg.drop(columns="target")), start=1):

                X_train = df_reg.iloc[train_idx].drop(columns="target").values
                y_train = df_reg.iloc[train_idx]["target"].values

                X_test = df_reg.iloc[test_idx].drop(columns="target").values
                y_test = df_reg.iloc[test_idx]["target"].values

                # train a model on each fold
                model = xgb.XGBRegressor()
                # model = KNeighborsRegressor(n_neighbors=1)

                # fit model with default hyperparameters, without validation set
                model.fit(X_train, y_train)
                # surrogate_blackbox = add_surrogate(blackbox, predict_curves=False)

                predictions = model.predict(X_test)
                # predictions = surrogate_blackbox(X_test, fidelity={'epoch': 0})

                plt.scatter(y_test, predictions, marker='+', label=f"Fold {fold}", alpha=0.7)

                rmse = root_mean_squared_error(y_test, predictions)
                rmse_per_fold.append(rmse)

                r2 = r2_score(y_test, predictions)
                r2_per_fold.append(r2)

                # print(f" Fold {fold} MSE: {rmse:.4f} (test rows={len(test_idx)})")

            plt.plot([0,1],[0,1], color='gray', linestyle=':')
            plt.xlabel("Real AUC_OVO")
            plt.ylabel("Predicted AUC_OVO")
            plt.title(f"{ae_architecture}: {datascenario}_{task}: Mean R2: {np.mean(r2_per_fold):.4f} ± {np.std(r2_per_fold):.4f}")
            plt.xlim([.4, 1])
            plt.ylim([.4, 1])
            plt.legend()
            plt.show()

            print(f'RUN: {ae_architecture} - {datascenario} - {task}',
                  f"Mean RMSE: {np.mean(rmse_per_fold):.4f} ± {np.std(rmse_per_fold):.4f}",
                  f"Mean R2: {np.mean(r2_per_fold):.4f} ± {np.std(r2_per_fold):.4f}")
