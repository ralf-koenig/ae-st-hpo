import pandas as pd

# needed to to_excel
import openpyxl


df= pd.read_parquet("data/raw/single_cell_seq_human_cortex/scATAC_human_cortex_clinical_formatted.parquet")

print(df.shape)

exit()

# Load input data into DataFrame
df1 = pd.read_parquet("real_ae_results_vanillix_tcga_RNA.parquet")
df2 = pd.read_parquet("real_ae_results_varix_tcga_RNA.parquet")
data = pd.concat([df1, df2])
data.to_excel("real_ae_results_tcga_RNA.xlsx")

df1 = pd.read_parquet("real_ae_results_vanillix_tcga_METH.parquet")
df2 = pd.read_parquet("real_ae_results_varix_tcga_METH.parquet")
data = pd.concat([df1, df2])
data.to_excel("real_ae_results_tcga_METH.xlsx")

df1 = pd.read_parquet("real_ae_results_vanillix_tcga_DNA.parquet")
df2 = pd.read_parquet("real_ae_results_varix_tcga_DNA.parquet")
data = pd.concat([df1, df2])
data.to_excel("real_ae_results_tcga_DNA.xlsx")

df1 = pd.read_parquet("real_ae_results_vanillix_schc_RNA.parquet")
df2 = pd.read_parquet("real_ae_results_varix_schc_RNA.parquet")
data = pd.concat([df1, df2])
data.to_excel("real_ae_results_schc_RNA.xlsx")

df1 = pd.read_parquet("real_ae_results_vanillix_schc_METH.parquet")
df2 = pd.read_parquet("real_ae_results_varix_schc_METH.parquet")
data = pd.concat([df1, df2])
data.to_excel("real_ae_results_schc_METH.xlsx")

