import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
# import diptest
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde, entropy
from datetime import datetime

import seaborn as sns

def dataframe_to_grayscale_image(df, save_path, log=False):
    array = df.to_numpy()
    array_min = np.nanmin(array)
    array_max = np.nanmax(array)
    if log:
        array = np.log1p(array)
        array_min = np.nanmin(array)
        array_max = np.nanmax(array)
        print("After normalization (log1p):", array_min, array_max)
    normalized = (2**16-1) * (array - array_min) / (array_max - array_min)
    normalized = np.nan_to_num(normalized)  # Replace NaNs with 0s (or adjust as needed)
    image_array = normalized.astype(np.uint16)
    image = Image.fromarray(image_array, mode='L')  # 'L' = grayscale
    image.save(save_path)


def plot_histogram( data, title, subtitle ):    # Create histogram
    plt.hist(data, bins=29, edgecolor='black')
    plt.title(f"""Histogram - {title} - {subtitle}""")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


def mean_of_means_and_skewness(df, subtitle):
    """
    Calculate the mean of means and mean of skewnesses of a Pandas DataFrame.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame with numeric columns.

    Returns:
    tuple: (mean_of_means, mean_of_skewnesses)
    """
    # numeric_df = df.select_dtypes(include='number')  # Ensure only numeric columns (int, float) are used
    means = df.mean()
    plot_histogram(means, "Means", subtitle)
    skewnesses = df.skew()
    plot_histogram(skewnesses, "Skewnesses", subtitle)
    return means.mean(), skewnesses.mean()


def diptest_histogram(df: pd.DataFrame):
    """
    Performs Hartigan's dip test for unimodality on each column of a DataFrame,
    then plots a histogram of the p-values.

    Parameters:
        df (pd.DataFrame): Input DataFrame with numeric columns.

    Returns:
        None. Displays a histogram of dip test p-values.
    """
    p_values = []

    for col in df.select_dtypes(include='number').columns:
        series = df[col].dropna().values
        if len(series) < 3:
            # Dip test needs more than 2 data points
            continue
        _, p_value = diptest.diptest(series)
        p_values.append(p_value)

    if not p_values:
        print("No valid numeric columns with sufficient data for dip test.")
        return

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(p_values, bins=20, edgecolor='black', color='skyblue')
    plt.title("Histogram of Dip Test p-values")
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_kl_divergence(empirical_samples, gmm, sample_size=1000):
    # Generate samples from GMM
    gmm_samples = gmm.sample(sample_size)[0].flatten()

    # Estimate densities
    kde_empirical = gaussian_kde(empirical_samples)
    kde_gmm = gaussian_kde(gmm_samples)

    # Evaluate densities at common points
    x_eval = np.linspace(min(empirical_samples.min(), gmm_samples.min()),
                         max(empirical_samples.max(), gmm_samples.max()), 1000)

    p = kde_empirical(x_eval)
    q = kde_gmm(x_eval)

    # Avoid division by zero
    p += 1e-10
    q += 1e-10

    return entropy(p, q)  # KL(p || q)


def gmm_equivalence(df, cols, max_components=10, kl_threshold=0.05):
    component_counts = {}

    for col in cols:
        data = df[col].dropna().values.reshape(-1, 1)
        # normalize data
        data = data / data.sum()
        if len(data) < 10:
            component_counts[col] = np.nan
            continue

        kl_list = []
        for n in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=n).fit(data)
            kl = compute_kl_divergence(data.flatten(), gmm)
            kl_list.append((n, kl))
            if kl < kl_threshold:
                component_counts[col] = n
                break
            if n == max_components:
                component_counts[col] = max_components

    # Plot histogram
    plt.figure(figsize=(8, 5))
    pd.Series(component_counts).dropna().astype(int).hist(bins=range(1, max_components + 2), align='left', rwidth=0.8)
    plt.xlabel('Number of GMM Components')
    plt.ylabel('Number of Columns')
    plt.title(f'Number of GMM Components needed for KL divergence (<{kl_threshold})')
    plt.grid(True)
    plt.show()

    return component_counts


def analyze_dataframe(df, most_variable_cols: int):

    # Compute statistics
    stats = pd.DataFrame({
        'Mean': df.mean(),
        'StdDev': df.std(),
        'Skewness': df.skew(),
        'Variance': df.var()
    })

    # Select top N most variable columns based on variance
    top_variable_cols = stats.sort_values('Variance', ascending=False).head(most_variable_cols)

    # Plot 1: Mean vs Standard Deviation
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=top_variable_cols['StdDev'], y=top_variable_cols['Mean'])
    plt.title(f'Top {most_variable_cols} Most Variable Columns: Mean vs Std')
    plt.ylabel('Mean')
    plt.xlabel('Standard Deviation')
    # for col in top_variable_cols.index:
    #     plt.text(top_variable_cols.loc[col, 'StdDev'], top_variable_cols.loc[col, 'Mean'], col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Skewness vs Standard Deviation
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=top_variable_cols['StdDev'], y=top_variable_cols['Skewness'])
    plt.title(f'Top {most_variable_cols} Most Variable Columns: Skewness vs Std')
    plt.ylabel('Skewness')
    plt.xlabel('Standard Deviation')
    # for col in top_variable_cols.index:
    #     plt.text(top_variable_cols.loc[col, 'StdDev'], top_variable_cols.loc[col, 'Skewness'], col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return stats

if __name__ == '__main__':
    print("THE CANCER GENOME ATLAS (TCGA)")

    print("""
    ############################################################################
    TCGA - Transcriptomics - RNA sequencing results""")

    # dataset = "bulk_seq_tcga/data_mrna_seq_v2_rsem" # diptest 12.000 near 1, about 3000 below 0.05
    # dataset = "bulk_seq_tcga/data_methylation_per_gene" # diptest 7500 near 1, 800 below 0.05
    # dataset = "bulk_seq_tcga/data_combi_MUT_CNA" # diptest all below .05
    # dataset = "bulk_seq_tcga/data_clinical"
    #
    dataset = "single_cell_seq_human_cortex/scRNA_human_cortex" # diptest all below .05
    # dataset = "single_cell_seq_human_cortex/scATAC_human_cortex" # diptest all below .05
    # dataset = "single_cell_seq_human_cortex/ scATAC_human_cortex_clinical"

    df = pd.read_parquet(f"data/raw/{dataset}_formatted.parquet")

    print(df.shape)
    print(df)
    print(df.stack().describe())

    # diptest_histogram(df)

    exit()

    nr_of_most_variable_cols = 2048

    analyze_dataframe(df, nr_of_most_variable_cols)

    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    analyze_dataframe(df_scaled, nr_of_most_variable_cols)

    analyze_dataframe(np.log1p(df), nr_of_most_variable_cols)

    df_scaled_log1p = pd.DataFrame(StandardScaler().fit_transform(np.log1p(df)), columns=df.columns)
    analyze_dataframe(df_scaled_log1p, nr_of_most_variable_cols)

    # exit()

    # use a random subsample of columns
    cols = df.sample(n=10, axis=1).columns
    print("Columns:", cols)

    print("Current Time:", datetime.now().strftime("%H:%M:%S"))
    result = gmm_equivalence(df, cols, kl_threshold=0.05)
    print(result)
    print ("Mean:", sum(result.values()) / len(result))

    print("Current Time:", datetime.now().strftime("%H:%M:%S"))
    result = gmm_equivalence(np.log1p(df), cols, kl_threshold=0.05)
    print ("Mean:", sum(result.values()) / len(result))

    print("Current Time:", datetime.now().strftime("%H:%M:%S"))

    # exit()

    mean_mean, mean_skew = mean_of_means_and_skewness(df, "TCGA RNA (original data)")
    print(f"Mean of Means (original data): {mean_mean}")
    print(f"Mean of Skewnesses (original data): {mean_skew}")

    # df_scaled = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    # mean_mean, mean_skew = mean_of_means_and_skewness(df_scaled, "TCGA RNA (scaled)")
    # print(f"Mean of Means (z-scaled): {mean_mean}")
    # print(f"Mean of Skewnesses (z-scaled): {mean_skew}")
    # confirmation: StandardScaler does not remove skew

    # log1p removes skew in the data
    mean_mean, mean_skew = mean_of_means_and_skewness(np.log1p(df), "TCGA RNA (log1p)")
    print(f"Mean of Means (log1p): {mean_mean}")
    print(f"Mean of Skewnesses (log1p): {mean_skew}")
    # applying finding: log 1p does reduce average skew considerably

    dataframe_to_grayscale_image(df, log=True, save_path="TCGA_RNA_log1p.png")

    print("""
    ############################################################################
    TCGA - Genomics - DNA data - mutations""")
    df = pd.read_parquet("data/raw/bulk_seq_tcga/data_combi_MUT_CNA_formatted.parquet")
    print(df)
    print(df.stack().describe())

    dataframe_to_grayscale_image(df, save_path="TCGA_DNA_Mutations_orig.png")

    mean_mean, mean_skew = mean_of_means_and_skewness(df, "TCGA DNA Mutations (original data)")
    print(f"Mean of Means (original data): {mean_mean}")
    print(f"Mean of Skewnesses (original data): {mean_skew}")

    # log1p removes skew in the data
    mean_mean, mean_skew = mean_of_means_and_skewness(np.log1p(df), "TCGA DNA Mutations (log1p)")
    print(f"Mean of Means (log1p): {mean_mean}")
    print(f"Mean of Skewnesses (log1p): {mean_skew}")
    # applying finding: log 1p does reduce average skew in this case

    dataframe_to_grayscale_image(df, log=True, save_path="TCGA_DNA_Mutations_log1p.png")

    threshold = 1
    print(f"Total entries above threshold({threshold}): {(df > threshold).sum().sum()}")

    print("""
    ############################################################################
    TCGA - Epigenomics - Methylation rates""")
    df = pd.read_parquet("data/raw/bulk_seq_tcga/data_methylation_per_gene_formatted.parquet")

    print(df)
    print(df.stack().describe())

    mean_mean, mean_skew = mean_of_means_and_skewness(df, "TCGA Methylation (original data)")
    print(f"Mean of Means (original data): {mean_mean}")
    print(f"Mean of Skewnesses (original data): {mean_skew}")

    dataframe_to_grayscale_image(df, "TCGA_Methylation.png")

    print("""
    ############################################################################
    SINGLE CELL from HUMAN CORTEX in different development stages
    ############################################################################
    """)

    print("""
    ############################################################################
    SINGLE CELL HUMAN CORTEX - Transcriptomics - RNA sequencing results""")

    df = pd.read_parquet("data/raw/single_cell_seq_human_cortex/scRNA_human_cortex_formatted.parquet")
    print(df)
    print(df.stack().describe())

    mean_mean, mean_skew = mean_of_means_and_skewness(df, "Single Cell Human Cortex - RNA (original data)")
    print(f"Mean of Means (original data): {mean_mean}")
    print(f"Mean of Skewnesses (original data): {mean_skew}")

    dataframe_to_grayscale_image(df, save_path="SC_Human_Cortex_RNA.png")

    print("""
    ############################################################################
    SINGLE CELL HUMAN CORTEX - Epigenomics- Methylation results
    """)
    df = pd.read_parquet("data/raw/single_cell_seq_human_cortex/scATAC_human_cortex_formatted.parquet")
    print(df)
    print(df.stack().describe())

    mean_mean, mean_skew = mean_of_means_and_skewness(df, "Single Cell Methylation (original data)")
    print(f"Mean of Means (original data): {mean_mean}")
    print(f"Mean of Skewnesses (original data): {mean_skew}")

    dataframe_to_grayscale_image(df, "SC_Human_Cortex_Methylation.png")

