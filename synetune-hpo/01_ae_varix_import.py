# Source:  main/syne_tune/blackbox_repository/README.md

# Add your own blackbox
# We assume that your data includes:
#
# a search space
# a list of hyperparameter configurations sampled from the search space + their performance metrics
# optionally: a set of evaluations for multiple fidelity levels

# To add your own blackbox, follow these steps:
#
# Implement a new class that is derived from BlackboxRecipe.

# Crucially, this class needs to implement the function _generate_on_disk(),
# which loads your original data and stores it in our format.
# For that, you can call the serialize() function, which expects a dictionary where the key specifies
# the task name and the value is a BlackboxTabular object.

# Next, add your new class to recipes.py and run the following command to generate your new blackbox.
# To avoid your upload to the Hugging Face repo from failing, set upload_on_hub=False.
#    python generate_and_upload_all_blackboxes.py

# Lastly, you can test your blackbox by running the following script:
#    python repository.py

# This file is adapted from syne-tune/blackbox-repository/conversion_scripts/scripts/lcbench.py

import pandas as pd
import numpy as np

from syne_tune.blackbox_repository.blackbox_tabular import serialize, BlackboxTabular
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts import (
    metric_elapsed_time,
    default_metric,
    time_attr,
)
from syne_tune.config_space import randint, lograndint, uniform, loguniform, choice
from syne_tune.util import catchtime

from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path

BLACKBOX_NAME = "autoencodix_varix" # create one blackbox per AE architecture, as it has its hyperparameter set

METRIC_DOWNSTREAM_PERFORMANCE = "WEIGHTED_AVG_AUC_DOWNSTREAM_PERFORMANCE"
METRIC_ELAPSED_TIME = "training_runtime"
TIME_ATTR = "epoch"

MAX_RESOURCE_LEVEL = 1

CONFIGURATION_SPACE = {
        # Model Parameters
        'K_FILTER': choice([128, 256, 512, 1024, 2048, 4096]), # 6 values
        'N_LAYERS': choice([2, 3, 4]), # 3 values
        'ENC_FACTOR': uniform(1, 4),
        'LATENT_DIM_FIXED': choice([2, 4, 8, 16, 32, 64]),  # 6 values
        # Training Parameters
        'BATCH_SIZE': choice([32, 64, 128, 256]), # 4 values
        # 'EPOCHS', # is constant in the runs at 300, nothing can be learnt from it
        'LR_FIXED': loguniform(1e-5, 1e-1),
        'DROP_P': uniform(0, 0.9),
        'BETA': loguniform(.001, 10)
        # overall: 6*3*6*4 = 432 values times values from 3 uniform/loguniform scales
        # with an assumption: 3 discrete values per uniform scale
        # 432 * 27 = 11.664 hyperparameter combinations
        # with an assumption: 5 discrete values per uniform scale
        # 432 * 125 = 54.000 hyperparameter combinations
}

class AutoEncodixVarixBlackboxRecipe(BlackboxRecipe):
    def __init__(self):
        super(AutoEncodixVarixBlackboxRecipe, self).__init__(
            name=BLACKBOX_NAME,
            cite_reference="Varix (VAE) from 'A generalized and versatile framework to train and evaluate autoencoders for biological representation learning and beyond: AUTOENCODIX',"
            "Maximilian Joas, Neringa Jurenaite, Dusan Prascevic, Nico Scherf, Jan Ewald. BioRxiv, 2024.",
        )

    def _generate_on_disk(self):

        with catchtime("loading data"):
            # load data frames with results from parquet files
            df_dict = {
                ('tcga','rna')  : pd.read_parquet("../autoencodix-hpo/real_ae_results_varix_tcga_RNA.parquet"),
                ('tcga','meth') : pd.read_parquet("../autoencodix-hpo/real_ae_results_varix_tcga_METH.parquet"),
                ('tcga', 'dna') : pd.read_parquet("../autoencodix-hpo/real_ae_results_varix_tcga_DNA.parquet"),
                ('schc', 'rna') : pd.read_parquet("../autoencodix-hpo/real_ae_results_varix_schc_RNA.parquet"),
                ('schc', 'meth'): pd.read_parquet("../autoencodix-hpo/real_ae_results_varix_schc_METH.parquet"),
            }

        with catchtime("converting"):
            hyperparameters_dict = {}
            for (data_scenario, task), df in df_dict.items():
                hyperparameters_dict[(data_scenario, task)] = df[[
                    # model parameters
                    "K_FILTER", "N_LAYERS", "ENC_FACTOR", "LATENT_DIM_FIXED",
                    # training parameters
                    "LR_FIXED", "BATCH_SIZE", "DROP_P", "BETA"]]

                # add weighted average, taking into account high correlation between AUC on CANCER_TYPE, SUB_TYPE, ONCOTREE_CODE
                if data_scenario == 'tcga':
                    df['WEIGHTED_AVG_AUC_DOWNSTREAM_PERFORMANCE'] = (
                                                              df['CANCER_TYPE'] * 1/21
                                                            + df['SUBTYPE']  * 1/21
                                                            + df['ONCOTREE_CODE'] * 1/21
                                                            + df['SEX'] * 1/7
                                                            + df['AJCC_PATHOLOGIC_TUMOR_STAGE'] * 1/7
                                                            + df['GRADE'] * 1/7
                                                            + df['PATH_N_STAGE'] * 1/7
                                                            + df['DSS_STATUS'] * 1/7
                                                            + df['OS_STATUS'] * 1/7 )
                elif data_scenario == 'schc':
                    df['WEIGHTED_AVG_AUC_DOWNSTREAM_PERFORMANCE'] = (df['author_cell_type'] * 1/3 +
                                                                     df['age_group'] * 1/3 +
                                                                     df['sex'] * 1/3 )

            objectives = [
                "training_runtime",
                "valid_r2",
                # TCGA: individual AUC results on clinical variables
                # 'CANCER_TYPE', 'SUBTYPE', 'ONCOTREE_CODE', 'SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE', 'GRADE',
                # 'PATH_N_STAGE', 'DSS_STATUS', 'OS_STATUS',
                # SCHC: individual AUC results on clinical variables
                # 'author_cell_type', 'age_group', 'sex'

                # Overall weighted average for single objective optimization later
                'WEIGHTED_AVG_AUC_DOWNSTREAM_PERFORMANCE'
            ]

            # use a constant value here for the moment, later use a range 1 to 300 for the epochs on valid_r2
            # this is naming the fidelities in a dict
            fidelity_space = {TIME_ATTR: randint(lower=300, upper=300)}

            bb_dict = {}

            for (data_scenario, task) in df_dict.keys():
                task_name = data_scenario+"_"+task
                # must be a numpy array
                objectives_evaluations = df_dict[(data_scenario, task)][objectives].to_numpy()
                # convert (x by y) to (x, 1, 1, y)
                objectives_evaluations = np.expand_dims(np.expand_dims(objectives_evaluations, axis=1), axis=2)

                bb_dict[task_name] = BlackboxTabular(
                    hyperparameters=hyperparameters_dict[(data_scenario, task)],
                    configuration_space=CONFIGURATION_SPACE,
                    fidelity_space=fidelity_space,  # this is a dict
                    objectives_evaluations=objectives_evaluations,
                    fidelity_values=np.arange(1),  # single [0] in a numpy array for one epoch value
                    objectives_names=objectives,
                )

        with catchtime("saving to disk"):
            serialize(
                bb_dict=bb_dict,
                path=repository_path / self.name,
                metadata={
                    metric_elapsed_time: METRIC_ELAPSED_TIME, # one of the time metrics
                    default_metric: METRIC_DOWNSTREAM_PERFORMANCE, #
                    time_attr: TIME_ATTR, # epochs
                },
            )


if __name__ == "__main__":
    AutoEncodixVarixBlackboxRecipe().generate(upload_on_hub=False)
