"""
Create YAML config files to run AutoEncodix on two data scenarios (TCGA, Single Cell Human Cortex)
and their tasks (which are the modalities).
"""

import yaml
import pandas as pd
from pathlib import Path

def get_random_config(ae_architecture, data_scenario, task, hyperparams):
    """
    Get a configuration for AutoEncodix that combines the hyperparameter combination
    with fix definitions for the dataset and task

    :return:
    config - a config dict
    """
    cfg = dict()
    # DATA DEFINITIONS ------------------------------------------------------------
    # -----------------------------------------------------------------------------
    cfg['DATA_TYPE'] = dict()
    # cfg['DATA_TYPE']['RNA']['FILE_RAW'] = "scRNA_human_cortex_formatted.parquet"
    # cfg['DATA_TYPE']['METH']['FILE_RAW'] = "stad_meth_formatted.parquet"
    # cfg['DATA_TYPE']['METH']['FILE_RAW'] = "data_methylation_per_gene_formatted.parquet"
    cfg['DATA_TYPE'][task] = dict()
    cfg['DATA_TYPE'][task]['TYPE'] = "NUMERIC"
    cfg['DATA_TYPE'][task]['FILTERING'] = "Var"  # We filter for feature with the highest variance
    cfg['DATA_TYPE'][task]['SCALING'] = "Standard"  # We scale features by the standard scaler
    if data_scenario == 'tcga':
        if task == 'RNA':
            cfg['DATA_TYPE'][task]['FILE_RAW'] = "data_mrna_seq_v2_rsem_formatted.parquet"
        elif task=='METH':
            cfg['DATA_TYPE'][task]['FILE_RAW'] = "data_methylation_per_gene_formatted.parquet"
        elif task=='DNA':
            cfg['DATA_TYPE'][task]['FILE_RAW'] = "data_combi_MUT_CNA_formatted.parquet"
    elif data_scenario == 'schc':
        if task == 'RNA':
            cfg['DATA_TYPE'][task]['FILE_RAW'] = "scRNA_human_cortex_formatted.parquet"
        elif task=='METH':
            cfg['DATA_TYPE'][task]['FILE_RAW'] = "scATAC_human_cortex_formatted.parquet"

    # Clinical Parameters for plotting
    cfg['DATA_TYPE']['ANNO'] = dict()
    cfg['DATA_TYPE']['ANNO']['TYPE'] = "ANNOTATION"
    if data_scenario == 'tcga':
        cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "data_clinical_formatted.parquet"
    elif data_scenario == 'schc':
        cfg['DATA_TYPE']['ANNO']['FILE_RAW'] = "scATAC_human_cortex_clinical_formatted.parquet"

    ## Model and Training --------------------------------------------------------
    # ----------------------------------------------------------------------------
    # Reproducibility
    cfg['FIX_RANDOMNESS'] = "random"
    # cfg['GLOBAL_SEED'] = 42
    # Model
    cfg['MODEL_TYPE'] = ae_architecture  # Train an AE of the type: plain AE = vanillix, VAE = varix
    cfg['TRAIN_TYPE'] = "train"  # simple training, no tuning
    cfg['K_FILTER'] = int(hyperparams['K_FILTER'])  # Input features per data modality
    cfg['N_LAYERS'] = int(hyperparams['N_LAYERS'])  # OPTIONAL number of layers for the encoder, respectively decoder (all integer values are possible)
    cfg['ENC_FACTOR'] = hyperparams['ENC_FACTOR']  # OPTIONAL Factor (float) by which the number of input neurons is divided to get the number of output neurons. Should be larger than 1 for compressing autoencoder.
    cfg['LATENT_DIM_FIXED'] = int(hyperparams['LATENT_DIM_FIXED']) # Latent space dimension
    cfg['RECONSTR_LOSS'] = "MSE"  # loss function for reconstruction
    cfg['VAE_LOSS'] = "KL"  # loss function distribution distance for Varix
    # Training
    cfg['EPOCHS'] = 300 # based on analysis of loss curves
    cfg['LR_FIXED'] = hyperparams['LR_FIXED'] # Learning rate
    cfg['BETA'] = hyperparams['BETA']  # weighting of VAE loss for Varix
        # also set BETA for vanillix due to bug in AutoEncodix that always requires BETA even if not needed/used by vanillix
    cfg['BATCH_SIZE'] = int(hyperparams['BATCH_SIZE'])
    cfg['DROP_P'] = hyperparams['DROP_P']
    # Prediction
    cfg['PREDICT_SPLIT'] = "all"  # Embedding of all samples should be calculated in prediction
    # EVALUATION and VISUALIZATION ------------------------------------------------
    # -----------------------------------------------------------------------------
    cfg['DIM_RED_METH'] = "UMAP"  # For 2D visualization when LATENT_DIM_FIXED>2

    if data_scenario == 'tcga':
        cfg['CLINIC_PARAM'] = [  # Parameters to colorize plots and perform embedding evaluation
            "CANCER_TYPE",
            "SUBTYPE",
            "ONCOTREE_CODE",
            "SEX",
            "AJCC_PATHOLOGIC_TUMOR_STAGE",
            "GRADE",
            "PATH_N_STAGE",
            # "TMB_NONSYNONYMOUS", # a regression task, not classification, therefore exclude
            # "AGE",               # a regression task, not classification, therefore exclude
            "DSS_STATUS",
            "OS_STATUS"
        ]
    elif data_scenario == 'schc':
        cfg['CLINIC_PARAM'] = [  # Parameters to colorize plots and perform embedding evaluation
            "author_cell_type",
            "age_group",
            "sex",
        ]

    cfg['ML_TYPE'] = "Auto-detect"  # Is CLINIC_PARAM prediction either regression or classification?
    cfg['ML_ALG'] = [  # ML algorithms for embedding evaluation
        'Linear',
        # 'RF'
    ]
    cfg['ML_SPLIT'] = "use-split"  # Test ML performance on train, test, valid split
    cfg['ML_TASKS'] = [  # Compare embeddings to other dimension reduction methods
        'Latent',
        'UMAP',
        'PCA',
        'RandomFeature'
    ]
    return cfg

def write_config_list_to_yaml( config_list ):
    """
        save configs in config list to individual YAML files
    """
    for cfg in config_list:
        target_path = CONFIG_RUNS_GENERATED_PATH / (cfg['RUN_ID'] + "_config.yaml")
        with target_path.open('w') as file:
            yaml.dump(cfg, file)

if __name__ == '__main__':
    """
    Create a list of config files for AutoEncodix and save them in YAML format
    """
    CONFIGS_START = 2500
    CONFIGS_STOP = 3000
    # CONFIG_RUNS_GENERATED_PATH =  Path('\\\\wsl$/Ubuntu/home/ralf/autoencodix/config_runs_generated')
    CONFIG_RUNS_GENERATED_PATH = Path('/home/ralf/autoencodix/config_runs_generated')

    hyperparam_configs_df = pd.read_parquet("hyperparam_configs_5000.parquet")

    for ae_architecture in ['vanillix', 'varix']:
        for (data_scenario, task) in [ ('tcga','RNA'), ('tcga','METH'), ('tcga','DNA'),
                                     ('schc','RNA'), ('schc','METH') ]:
            config_list = []
            for i in range(CONFIGS_START, CONFIGS_STOP):
                cfg = get_random_config(ae_architecture,
                                        data_scenario,
                                        task,
                                        hyperparam_configs_df.iloc[i].to_dict()
                                        )
                cfg['RUN_ID'] = f"RUN_{ae_architecture}_{data_scenario}_{task}_{i:06d}" # string with leading zeros
                config_list.append(cfg)
            write_config_list_to_yaml(config_list)
