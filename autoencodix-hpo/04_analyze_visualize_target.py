import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Setup ---

# Create output directory for plots
# output_dir = "plots"
output_dir = "01_tcga_rna_common_hp"
os.makedirs(output_dir, exist_ok=True)

# Load input data into DataFrame
df1 = pd.read_parquet("real_ae_results_vanillix_tcga_RNA.parquet")
df2 = pd.read_parquet("real_ae_results_varix_tcga_RNA.parquet")
data = pd.concat([df1, df2])
# data = pd.read_parquet("03_ae_tcga_dna_500_results.parquet")
# data = pd.read_parquet("ae_sc_human_cortex_rna_500_meth_500.parquet")

# For split of SC data
# First 500 rows, on RNA data
# data = data.iloc[:500]
# Second 500 rows on Methylation
# data = data.iloc[500:1000]

records_before_filtering = data.shape[0]

# filter outliers on certain columns, these were determined manually by visual inspection
# filter_variables = [
#     'data_runtime', 'training_runtime', 'predicting_runtime', 'visualizing_runtime', 'total_runtime',
#     'train_recon_loss', 'valid_recon_loss', 'train_r2', 'valid_r2',
#     'CANCER_TYPE', 'AJCC_PATHOLOGIC_TUMOR_STAGE'
# ]

# # Calculate 1st and 99th percentiles
# filter_variables = ['training_runtime', 'total_runtime', 'train_r2', 'valid_r2']
# lower = data[filter_variables].quantile(0.01)
# upper = data[filter_variables].quantile(0.99)
# # Apply the filter: keep rows where all numeric values are within [1st, 99th] percentiles
# filtered_df = data[
#     (data[filter_variables] >= lower).all(axis=1) &
#     (data[filter_variables] <= upper).all(axis=1) ]
# data = filtered_df
# records_after_filtering = data.shape[0]
# print(f"Filtered {records_before_filtering - records_after_filtering} records for outliers (outside 1-99%).")


np.random.seed(42)
input_features = [
    # 'run_id', 'MODEL_TYPE',
    'K_FILTER',
    "N_LAYERS",
    'ENC_FACTOR',
    'LATENT_DIM_FIXED',
    # 'BATCH_SIZE', # kept constant
    'LR_FIXED',
    # 'BETA',
    'DROP_P',
    'trainable_parameters'
]

target_variables = [
    # runtime results
    'data_runtime', 'training_runtime', 'predicting_runtime',
    'visualizing_runtime', 'ml_task_runtime', 'total_runtime',
    'trainable_parameters',
    # loss results
    # 'train_recon_loss', # 'train_vae_loss',
    'train_r2',
    # 'train_total_loss',

    # 'valid_recon_loss', # 'valid_vae_loss',
    'valid_r2',
    # 'valid_total_loss',

    # AUC results
    'CANCER_TYPE',
    # 'SUBTYPE',
    # 'ONCOTREE_CODE',
    'SEX',
    'AJCC_PATHOLOGIC_TUMOR_STAGE',
    # 'GRADE',
    # 'PATH_N_STAGE',
    'DSS_STATUS',
    'OS_STATUS'
    # For single cell human cortex data
    # 'age_group', 'sex', 'author_cell_type'
]

# Separate inputs and targets
X = data[input_features]
Y = data[target_variables]

# --- Descriptive Statistics ---

# 1. Input Features
print("Input Features Description:\n", X.describe().T)
print("\nInput Features Skewness:\n", X.skew())
print("\nInput Features Kurtosis:\n", X.kurtosis())

# 2. Target Variables (grouped)
target_groups = [
    ['data_runtime', 'training_runtime', 'predicting_runtime',
    'visualizing_runtime', 'ml_task_runtime', 'total_runtime'
    ],
    [
        # 'train_recon_loss',
        # 'train_vae_loss',
        'train_r2',
        # 'train_total_loss',
        # 'valid_recon_loss',
        # 'valid_vae_loss',
        'valid_r2',
        # 'valid_total_loss'
    ],
    [
        'CANCER_TYPE',
        # 'SUBTYPE',
        # 'ONCOTREE_CODE',
        'SEX',
        'AJCC_PATHOLOGIC_TUMOR_STAGE',
        # 'GRADE',
        # 'PATH_N_STAGE',
        'DSS_STATUS',
        'OS_STATUS'

        # 'author_cell_type',
        # 'age_group',
        # 'sex',
    ]
]

for idx, group in enumerate(target_groups):
    print(f"\nTarget Group {idx+1} Description:")
    print(Y[group].describe().T)
    print("\nSkewness:\n", Y[group].skew())
    print("\nKurtosis:\n", Y[group].kurtosis())

# --- Visualization Functions ---

def plot_distribution(df, title_prefix="", save_prefix=""):
    """Plot histograms and density plots for a DataFrame."""
    n_cols = 3
    n_rows = (len(df.columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    print("Plotting distribution")
    for idx, col in enumerate(df.columns):
        sns.histplot(df[col],
                     # kde=True,
                     ax=axes[idx], color="skyblue")
        axes[idx].set_title(f"{title_prefix}{col}")

    for i in range(len(df.columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    print("Saving distribution")
    save_path = os.path.join(output_dir, f"{save_prefix}_distribution.png")
    plt.savefig(save_path)
    plt.close()

def plot_boxplots(df, title_prefix="", save_prefix=""):
    """Plot boxplots for a DataFrame."""
    print("Building boxplots")
    n_cols = 3
    n_rows = (len(df.columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for idx, col in enumerate(df.columns):
        sns.boxplot(y=df[col], ax=axes[idx], color="lightgreen")
        axes[idx].set_title(f"{title_prefix}{col}")

    for i in range(len(df.columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    print("Saving boxplots")
    save_path = os.path.join(output_dir, f"{save_prefix}_boxplots.png")
    plt.savefig(save_path)
    plt.close()

def plot_correlation_heatmap(df, title="Correlation Matrix", save_prefix=""):
    """Plot a correlation heatmap."""
    plt.figure(figsize=(10, 8))
    corr = df.corr(method='spearman', numeric_only=True)
    print("Building ClusterMap")
    sns.clustermap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    print("Saving ClusterMap")
    save_path = os.path.join(output_dir, f"{save_prefix}_correlation.png")
    plt.savefig(save_path)
    plt.close()

def plot_pairplot(df, title_prefix="", save_prefix=""):
    """Plot pairplot for a DataFrame."""
    print("Building Pairplots")
    pp = sns.pairplot(df, corner=True, plot_kws=dict(alpha=.15), hue='MODEL_TYPE')
    log_columns = [# "BETA",
                   "LR_FIXED",
                   'trainable_parameters'
    ]
    for ax in pp.axes.flat:
        if (ax is not None) and (ax.get_xlabel() in log_columns):
            ax.set(xscale="log")
    plt.suptitle(f"{title_prefix} Pairplot", y=1.02)
    print("Saving Pairplots")
    save_path = os.path.join(output_dir, f"{save_prefix}_pairplot.png")
    plt.savefig(save_path)
    plt.close()

# --- Plotting ---

# 1. Input Features
print("\nPlotting Input Features...")
plot_distribution(X, title_prefix="Input - ", save_prefix="input_features")
plot_boxplots(X, title_prefix="Input - ", save_prefix="input_features")
plot_correlation_heatmap(X, title="Input Features Correlation", save_prefix="input_features")
plot_pairplot(data[input_features + ['MODEL_TYPE']], title_prefix="Input Features", save_prefix="input_features")

# 2. Target Variables by Group
for idx, group in enumerate(target_groups):
    print(f"\nPlotting Target Group {idx+1}...")
    save_prefix = f"target_group_{idx+1}"
    plot_distribution(Y[group], title_prefix=f"Target Group {idx+1} - ", save_prefix=save_prefix)
    plot_boxplots(Y[group], title_prefix=f"Target Group {idx+1} - ", save_prefix=save_prefix)
    plot_correlation_heatmap(data[input_features + group], title=f"Target Group {idx+1} Correlation (Spearman)", save_prefix=save_prefix)
    plot_pairplot(
        data[input_features + group + ['MODEL_TYPE']],
        title_prefix=f"Target Group {idx+1}",
        save_prefix=save_prefix
    )

print(f"\nAll plots saved to '{output_dir}/' folder.")
