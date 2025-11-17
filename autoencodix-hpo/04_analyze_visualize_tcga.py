import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    # df = pd.read_parquet("ae_runs_results_2000_scl.parquet.off")
    # df = pd.read_parquet("ae_tcga_rna_500_results.parquet")
    df1 = pd.read_parquet("real_ae_results_vanillix_tcga_RNA.parquet")
    df2 = pd.read_parquet("real_ae_results_varix_tcga_RNA.parquet")
    df = pd.concat([df1, df2])

    # Analyze correlation of AUC classification results
    METHOD = 'pearson' # pearson or spearman

    # Plot standard correlation Pearson (default) correlation or Spearman rank correlation (with method='spearman')
    g = sns.clustermap( df[[
        'CANCER_TYPE', 'SUBTYPE', 'ONCOTREE_CODE',
        'SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE', 'GRADE', 'PATH_N_STAGE',
         'DSS_STATUS', 'OS_STATUS']].corr(method=METHOD),
                vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f",
                cmap='coolwarm')
    g.fig.suptitle(f"Correlation Matrix ({METHOD} corr. coefficient)\n"
                   f"(TCGA RNA Classification Results, 3000+3000 runs)")
    plt.show()

    # Analyze correlations
    # plt.figure(figsize=(16, 12))
    # Plot Pearson correlation (would be default)
    # or Spearman (with method=spearman)

    # Analyze correlation of AUC classification results
    g = sns.clustermap( df[[
        'K_FILTER', 'N_LAYERS', 'ENC_FACTOR', 'LATENT_DIM_FIXED',
        'trainable_parameters',
        # 'BATCH_SIZE',
        'LR_FIXED', 'BETA', 'DROP_P',
        'data_runtime', 'training_runtime', 'predicting_runtime',
        'visualizing_runtime', 'ml_task_runtime', 'total_runtime',
        'train_recon_loss',
        # 'train_vae_loss',
        'train_r2', 'train_total_loss',
        'valid_recon_loss',
        # 'valid_vae_loss',
        'valid_r2', 'valid_total_loss',
        'CANCER_TYPE', 'SUBTYPE', 'ONCOTREE_CODE',
        'SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE', 'GRADE',
        'PATH_N_STAGE', 'DSS_STATUS', 'OS_STATUS'
    ]].corr(method=METHOD),
                vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={'size': 5},
                cmap='coolwarm')
    g.fig.suptitle(f"Correlation Matrix ({METHOD} corr. coefficient)\n"
                   f"(TCGA RNA Classification Results, 3000+3000 runs)")
    plt.show()

    # Sort the values and compute CDF
    for column in ['CANCER_TYPE',
                   'train_total_loss', 'train_r2',
                   'valid_total_loss', 'valid_r2',
                   'training_runtime', 'total_runtime'
                   ] :

        # Plot the CDF
        sorted_vals = df[column].sort_values()
        cdf = sorted_vals.rank(method='first') / len(sorted_vals)
        plt.plot(sorted_vals, cdf, marker='.', linestyle='none')
        plt.xlabel(column)
        plt.ylabel('CDF')
        plt.title('CDF Plot on ' + column)
        plt.grid(True)
        if column in ['train_total_loss', 'valid_total_loss']:
            plt.xlim(50, 1e4)
            plt.xscale('log')
        elif column in ['training_runtime', 'total_runtime']:
            plt.xlim(0, 4000)
        elif column in ['train_r2', 'valid_r2']:
            plt.xlim(-.2, .8)
        if column in ['LR_FIXED', 'BETA']:
            plt.xscale('log')
        plt.show()
