"""
Gather the results of previous runs of AutoEncodix into a data frame
for better analysis.
"""

from pathlib import Path
import yaml
import pandas as pd
# requires openpyxl in Excel writer to_excel

import re
from datetime import datetime

def get_logs (log_directory, ae_architecture, data_scenario, task):
    """
    Read log files from previous runs of AutoEncodix into a dict of list of lines.
    """
    log_dict = {}
    log_files = list(log_directory.rglob(f"RUN_{ae_architecture}_{data_scenario}_{task}_*/slurm*.out"))
    for file_name in log_files:
        run_id = file_name.relative_to(log_directory).parts[0]
        with open(file_name, 'r') as file:
            logfile = file.readlines()
            log_dict[run_id] = logfile
    return log_dict

def find_first_timestamp(lines, start_index, step):
    """
    Find first timestamp in log files starting from start_index in the direction of step parameter.

    :param lines: lines of a log file
    :param start_index: index of the line to start searching for a time stamp
    :param step:  direction in which to proceed from start_index
    :return: a time stamp in datetime format
    """
    # compile a pattern for YYYY-MM-DD hh:mm:ss,mmm
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
    index = start_index
    while index < len(lines):
        match = pattern.match(lines[index])
        if match:
            timestamp_str = match.group(1)
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
        index += step
    return None  # if no time stamp was found

def find_line_index(lines, search_text):
    """
    Find first index of a line in a list of log file lines that matches search_text.

    :param lines:
    :param search_text:
    :return: line index
    """
    pattern = re.compile(search_text)
    for index, line in enumerate(lines):
        if pattern.match(line):
           return index
    return None # if not found at all

def get_number_of_trainable_parameters_from_log (log):
    # example: Trainable params: 33,607,688
    index = find_line_index(log, "^Trainable params:")
    return int(re.sub(r'\D', '', log[index]))  # Remove non-digit characters and convert to int

def get_information_from_log (log):
    # start
    # done config
    # done data
    # Done training
    # Done predicting
    # Done visualizing
    # Done ml_task
    # end

    # get start time stamp from first line with time stamp
    # data_start = find_first_timestamp(log, find_line_index(log, "^done config"), 1 )
    data_start = find_first_timestamp(log, find_line_index(log, "✓ Configuration complete"), 1)
    # get end time stamp from last line
    data_end = find_first_timestamp(log, find_line_index(log, "^done data"), -1)
    data_runtime = (data_end-data_start).total_seconds()

    training_end = find_first_timestamp(log, find_line_index(log, "^Done training"), -1)
    training_runtime = (training_end - data_end).total_seconds()

    predicting_end = find_first_timestamp(log, find_line_index(log, "^Done predicting"), -1)
    predicting_runtime = (predicting_end - training_end).total_seconds()

    visualizing_end = find_first_timestamp(log, find_line_index(log, "^Done visualizing"), -1)
    visualizing_runtime = (visualizing_end - predicting_end).total_seconds()

    ml_task_end = find_first_timestamp(log, find_line_index(log, "^Done ml_task"), -1)
    ml_task_runtime = (ml_task_end - visualizing_end).total_seconds()

    total_end = find_first_timestamp(log, len(log)-1, -1)
    total_runtime =  (total_end-data_start).total_seconds()

    trainable_parameters = get_number_of_trainable_parameters_from_log(log)

    return [ data_runtime, training_runtime, predicting_runtime, visualizing_runtime, ml_task_runtime, total_runtime,
             trainable_parameters
    ]


def get_configs(config_directory, ae_architecture, data_scenario, task):
    """
    Get hyperparameter configurations from YAML files of previous AutoEncodix runs.
    :return:
    dict of dicts
    """
    config_dict = {}
    yaml_files = list(config_directory.rglob(f"RUN_{ae_architecture}_{data_scenario}_{task}_*/*.yaml"))

    for file_name in yaml_files:
        run_id = file_name.stem.replace("_config", "")
        with open(file_name, 'r') as file:
            cfg = yaml.safe_load(file)
            config_dict[run_id] = cfg
    return config_dict

def get_relevant_parameters_from_config ( cfg ):
    return [
        # AE architecture
        cfg['MODEL_TYPE'],
        # Model hyperparameters
        cfg['K_FILTER'], cfg['N_LAYERS'], cfg['ENC_FACTOR'], cfg['LATENT_DIM_FIXED'],
        # Training hyperparameters
        cfg['BATCH_SIZE'], cfg['EPOCHS'], cfg['LR_FIXED'], cfg['BETA'], cfg['DROP_P']
    ]

def get_loss_results(loss_results_directory, ae_architecture, data_scenario, task):
    """
    Get loss results from previous AutoEncodix runs.
    """
    row_labels = []
    values = []
    loss_files = list(loss_results_directory.rglob(f"RUN_{ae_architecture}_{data_scenario}_{task}_*/losses_*.parquet"))
    for file_name in loss_files:
        run_id = file_name.relative_to(loss_results_directory).parts[0]
        row_labels.append(run_id)
        df_losses = pd.read_parquet(file_name)
        column_labels = df_losses.columns.values
        # get the results of the last epoch
        values_last_row = df_losses.iloc[[-1]].values.flatten().tolist()
        values.append(dict(zip(column_labels, values_last_row)))

    df_loss_results = pd.DataFrame(values)
    df_loss_results['run_id'] = row_labels
    return df_loss_results

def get_ml_task_performance_results(ml_task_results_directory, ae_architecture, data_scenario, task):
    """
    Get ML performance results on downstream tasks from previous AutoEncodix runs.
    """
    row_labels = []
    column_labels = []
    values = []

    ml_task_result_files = list(
        ml_task_results_directory.rglob(f"RUN_{ae_architecture}_{data_scenario}_{task}_*/ml_task_performance.txt")
    )
    for file_name in ml_task_result_files:
        # run_id is the highest level directory name
        run_id = file_name.relative_to(ml_task_results_directory).parts[0]
        row_labels.append(run_id)
        with (open(file_name, 'r') as file):
            df_ml_results = pd.read_csv(file, sep='\t', header=0)
            # roc_auc_ovo_results_pca = df_ml_results[(df_ml_results.score_split=='test')
            #                                & (df_ml_results.ML_TASK=='PCA')
            #                                & (df_ml_results.metric=='roc_auc_ovo')]['value']
            df_roc_auc_ovo_results_latent = df_ml_results[(
                                                df_ml_results.score_split=='test') # results on test set
                                             & (df_ml_results.ML_TASK=='Latent') # results on Latent space variables
                                             & (df_ml_results.ML_ALG=='Linear') # results on Logistic Regression als GLM
                                             & (df_ml_results.metric=='roc_auc_ovo')] # classification metric
                # Area under Curve, Receiver Operating Characteristic, One vs. One for Multi-Class Classification
                # on the classification results on embeddings (latent space representations)
            column_labels = df_roc_auc_ovo_results_latent["CLINIC_PARAM"].to_numpy()
            values.append(df_roc_auc_ovo_results_latent["value"].to_list())

    df_ml_task_results = pd.DataFrame(values, columns=column_labels)
    df_ml_task_results['run_id'] = row_labels
    return df_ml_task_results


if __name__ == '__main__':
    RESULTS_INPUT_PATH = Path("/home/ralf/autoencodix/reports")

    for ae_architecture in ['vanillix', 'varix']:
        for (data_scenario, task) in [ ('tcga','RNA'), ('tcga','METH'), ('tcga','DNA'),
                                     ('schc','RNA'), ('schc','METH') ]:

            print(f"\nRunning {ae_architecture}_{data_scenario}_{task}:")
            print("Getting YAML configurations of previous runs of AutoEncodix.")
            configs = get_configs(RESULTS_INPUT_PATH, ae_architecture, data_scenario, task)
            hyperparameters_dict = {}
            for run_id, cfg in configs.items():
                relevant_parameters = get_relevant_parameters_from_config(cfg)
                hyperparameters_dict[run_id]=relevant_parameters
            df_hyperparameters = pd.DataFrame(hyperparameters_dict).T
            df_hyperparameters.columns = [
                'MODEL_TYPE', 'K_FILTER', 'N_LAYERS', 'ENC_FACTOR', 'LATENT_DIM_FIXED', 'BATCH_SIZE',
                'EPOCHS', 'LR_FIXED', 'BETA', 'DROP_P' ]
            df_hyperparameters = df_hyperparameters.reset_index().rename(columns={'index': 'run_id'})

            print("Getting runtime logs of previous runs of AutoEncodix.")
            logs = get_logs(RESULTS_INPUT_PATH, ae_architecture, data_scenario, task)
            runtimes_dict = {}
            for run_id, log in logs.items():
                # print(run_id)
                runtimes = get_information_from_log(log)
                runtimes_dict[run_id] = runtimes
            df_runtimes = pd.DataFrame(runtimes_dict).T
            df_runtimes.columns = ["data_runtime", "training_runtime", "predicting_runtime", "visualizing_runtime",
                                   "ml_task_runtime", "total_runtime", "trainable_parameters"]
            df_runtimes = df_runtimes.reset_index().rename(columns={'index': 'run_id'})

            print("Getting result of losses of previous runs of AutoEncodix.")
            df_loss_results = get_loss_results(RESULTS_INPUT_PATH, ae_architecture, data_scenario, task)

            print("Getting result of downstream ML task of previous runs of AutoEncodix.")
            df_ml_task_results = get_ml_task_performance_results(RESULTS_INPUT_PATH,
                                                                 ae_architecture, data_scenario, task)

            # assemble results into a dataframe for analysis, merging them side by side
            dfs = [df_hyperparameters, df_runtimes, df_loss_results, df_ml_task_results]
            df_merged = pd.concat([df.set_index(['run_id']) for df in dfs], axis=1).reset_index()
            df_merged.sort_values(by=['run_id'], ascending=True, inplace=True)
            df_merged.reset_index(drop=True, inplace=True)
            # print(df_merged)

            # save data frame to files
            df_merged.to_parquet(f"real_ae_results_{ae_architecture}_{data_scenario}_{task}.parquet")
