Running many hyperparameter combinations on Autoencodix 
========================================================

Files and directories are listed in workflow order.

## Input data
Original Parquet files from Ewald 2024 reproducibility paper and repository.
There it can be received by `get_preprocessed_data.sh`.
Source: https://zenodo.org/records/13691753

	Folder: data/raw/bulk_seq_tcga/

    Files TCGA:
	data_mrna_seq_v2_rsem_formatted.parquet
	data_methylation_per_gene_formatted.parquet
	data_combi_MUT_CNA_formatted.parquet
	data_clinical_formatted.parquet

	Folder: data/raw/single_cell_seq_human_cortex/

    Files SCHC:
	scRNA_human_cortex_formatted.parquet
	scATAC_human_cortex_formatted.parquet
	scATAC_human_cortex_clinical_formatted.parquet
	

## (optional) (Quantized) Pictures of input data

### Visualization of raw data

`00_visualize_raw_data.py` reads input data from Parquet files (float 32 bit) and saves them to greyscale images (8 bit) for a visual interpretation. This automatically implies: min-max-scaling to 0-255.

These images serve as visualization only. They are not used in the data workflow anywhere.
Beware: Artifacts from this quantization (called banding/posterization) can occur.

	Output Folder: data_as_images/

High-resolution with original number of features and samples (426 MB total), good for zooming in:

    Files:
	TCGA_RNA_log1p.png
	TCGA_Methylation.png
	TCGA_DNA_Mutations_log1p.png

	SC_Human_Cortex_RNA.png
	SC_Human_Cortex_Methylation.png

## Creation of 5000 hyperparameter configurations

`00_common_hpo_paramset.py` fixes 5000 hyperparameter configurations does random sampling from a hyperparamter configuration space and saves them to a Parquet file `hyperparam_configs_5000.parquet` to fix them. 

Fixed random hyperparameter combinations are important to later use them for all tasks in Syne Tune, as a blackbox needs common HP combinations across all tasks of a blackbox.
The set `hyperparam_configs_5000.parquet` is provided, that was used to create 3000 x 2 x 5 runs. 

	00_common_hpo_paramset.py
	hyperparam_configs_5000.parquet

`00_diptest_ralf.py` is an optional diptest (Hartigan & Hartigan's dip test of unimodality or multimodality in a data series). Was used to analze the data. Could be turned into dataset features, but I did not do that.

## Creation of the YAML files for Autoencodix

`01_create_yaml_for_data_scenario_tasks.py` reads the hyperparam_configs_5000.parquet and creates the 3000*10 YAML files for the 5 datasets (TCGA RNA, METH, DNA; SCHC RNA, METH) and 2 autoencoder architectures (Vanillix, Varix). 

These YAML files are then to be run on Autoencodix on a GPU cluster. There, it creates many files: YAML, ML_performance in CSV format, parquet files for losses and log files, where timing information can be extracted.

`f02_gather_results.py` file reads all the results from the Autoencodix runs: YAML for the configuration, ML_performance in CSV format, parquet files for losses and log files to extra timing information. It collects this into a Pandas dataframe and saves the results in Parquet format.

## Results from 30,000 runs on Autoencodix

Each file has 3000 records. 1 record for each Autoencodix run with the results from `f02_gather_results.py`.

Inside `ae_results_30000_runs/ae_results_30000_runs.zip` : 

### Vanillix on TCGA datasets

	real_ae_results_vanillix_tcga_RNA.parquet
	real_ae_results_vanillix_tcga_METH.parquet
	real_ae_results_vanillix_tcga_DNA.parquet

### Vanillix on SCHC datasets

	real_ae_results_vanillix_schc_RNA.parquet
	real_ae_results_vanillix_schc_METH.parquet

### Varix on TCGA datasets

	real_ae_results_varix_tcga_RNA.parquet
	real_ae_results_varix_tcga_METH.parquet
	real_ae_results_varix_tcga_DNA.parquet

### Varix on SCHC datasets

	real_ae_results_varix_schc_RNA.parquet
	real_ae_results_varix_schc_METH.parquet

### Excel export

`02a_results_to_excel.py`exports the Parquet files to (raw) Excel format.

## Loss Curve Evaluation

`03_evaluate_loss_curves.py` reads the loss files from the Autoencodix runs and then performs an analysis, when loss stabilizes. It was used to set a number of epochs (300), where results are mostly stable. The aim was that is epoch number is low to not so many compute ressources.

## Results Analysis

`04_analyze_visualize_target.py` filen the uses the Parquet files of the Autoencodix results written before. It created PairPlots, Correlation plots with optional filtering for outliers that would spoil the diagramms. It works with three sets of target group:
	* target variables in group 1: various runtimes.
	* target variables in group 2: various losses.
	* target variables in group 3: various values in ML downstream task performance.

	Output Folder: 01_tcga_rna_common_hp_filtered
	Putput Folder: 01_tcga_rna_common_hp_non_filtered

`04_analyze_visualize_tcga.py` does data-set specific analysis and plotting.

`05_plot_varix_sweep.py` plots a line plot with errorbars for Varix results.
Input data are 5 repeated runs and the ML downstream task results in a specific folder.

## Optional conda environment YAML file

`autoencodix-hpo-env.yaml` collects all the requirements.
