from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from f02_gather_results import get_ml_task_performance_results

def b_to_float(s: str) -> float:
    """
    Convert 'B' + digits into a float, where leading zeros in the digits
    indicate decimal scaling.
    """
    conv={ "B0001": 0.001, "B001":  0.01, "B003":  0.03, "B01":   0.1,
           "B03":   0.3,   "B1":    1.0,  "B3":    3.0,  "B10":  10.0 }
    return conv[s]

if __name__ == '__main__':
    for ae_architecture in ['varix']:
        for (data_scenario, task) in [ ('schc','RNA') ]:
            print(f"\nRunning {ae_architecture}_{data_scenario}_{task}:")
            fig, ax = plt.subplots(figsize=(9, 6))

            print("Getting result of downstream ML task of previous runs of AutoEncodix.")
            df_1 = get_ml_task_performance_results(Path("/home/ralf/autoencodix/reports_varix/varix_sweep_1"),
                                                                 ae_architecture, data_scenario, task)
            df_2 = get_ml_task_performance_results(Path("/home/ralf/autoencodix/reports_varix/varix_sweep_2"),
                                                                 ae_architecture, data_scenario, task)
            df_3 = get_ml_task_performance_results(Path("/home/ralf/autoencodix/reports_varix/varix_sweep_3"),
                                                                 ae_architecture, data_scenario, task)
            df_4 = get_ml_task_performance_results(Path("/home/ralf/autoencodix/reports_varix/varix_sweep_4"),
                                                                 ae_architecture, data_scenario, task)
            df_5 = get_ml_task_performance_results(Path("/home/ralf/autoencodix/reports_varix/varix_sweep_5"),
                                                                 ae_architecture, data_scenario, task)

            # add average downstream performance in new column
            for df in [df_1, df_2, df_3, df_4, df_5]:
                df["avg_ds_performance"] = (df["author_cell_type"] + df["age_group"] + df["sex"] ) / 3

            colors_per_lat={64: 'brown', 32: 'blue', 16: 'green', 8: 'cyan', 4: 'magenta', 2: 'red'}
            for lat in [64, 32, 16, 8, 4, 2]:
                x, y1, y2, y3, y4, y5 = [], [], [], [], [], []
                for beta in ["B0001", "B001", "B003", "B01", "B03", "B1", "B3", "B10"]:
                    x.append(b_to_float(beta))
                    y1.append(df_1[df_1["run_id"]=='RUN_varix_schc_RNA_ST_TL_'+beta+f'_Lat{lat}']["avg_ds_performance"].values[0])
                    y2.append(df_2[df_2["run_id"]=='RUN_varix_schc_RNA_ST_TL_'+beta+f'_Lat{lat}']["avg_ds_performance"].values[0])
                    y3.append(df_3[df_3["run_id"]=='RUN_varix_schc_RNA_ST_TL_'+beta+f'_Lat{lat}']["avg_ds_performance"].values[0])
                    y4.append(df_4[df_4["run_id"]=='RUN_varix_schc_RNA_ST_TL_'+beta+f'_Lat{lat}']["avg_ds_performance"].values[0])
                    y5.append(df_5[df_5["run_id"]=='RUN_varix_schc_RNA_ST_TL_'+beta+f'_Lat{lat}']["avg_ds_performance"].values[0])
                ax.plot(x, y1, "-s", color=colors_per_lat[lat])
                ax.plot(x, y2, "--*", color=colors_per_lat[lat])
                ax.plot(x, y3, "-.x", color=colors_per_lat[lat])
                ax.plot(x, y4, ":+", color=colors_per_lat[lat])
                ax.plot(x, y5, ":o", color=colors_per_lat[lat], label=f"{lat} lat dim's")

            ax.set_xscale("log")
            plt.title('SCHC RNA scenario - Varix with Beta sweep example in varying latent dimensions\nResults of true Autoencodix runs (n=5)')
            ax.set_xlabel("beta")
            ax.set_ylabel("avg downstream performance (cell type, age group, sex)")
            ax.set_ylim(0.65, 1)
            plt.legend()
            plt.show()

            fig, ax = plt.subplots(figsize=(9, 6))

            for lat in [64, 32, 16, 8, 4, 2]:
                x_vals, means, stds = [], [], []

                for beta in ["B0001", "B001", "B003", "B01", "B03", "B1", "B3", "B10"]:
                    x_val = b_to_float(beta)
                    x_vals.append(x_val)

                    # Collect all sweep values for this beta and latent dimension
                    values = []
                    for df in [df_1, df_2, df_3, df_4, df_5]:
                        val = df[df["run_id"] == f'RUN_varix_schc_RNA_ST_TL_{beta}_Lat{lat}'][
                            "avg_ds_performance"].values[0]
                        values.append(val)

                    # Compute mean and std for error bar
                    means.append(np.mean(values))
                    stds.append(np.std(values))

                # Plot error bars
                ax.errorbar(x_vals, means, yerr=stds, fmt='-o', color=colors_per_lat[lat],
                            capsize=5, label=f"{lat} lat dim's (mean ± std)")

            ax.set_xscale("log")
            plt.title(
                'SCHC RNA scenario - Varix with Beta sweep example in varying latent dimensions\nResults of true Autoencodix runs (n=5)')
            ax.set_xlabel("beta")
            ax.set_ylabel("avg downstream performance (cell type, age group, sex)")
            ax.set_ylim(0.65, 1)
            plt.legend()
            plt.show()