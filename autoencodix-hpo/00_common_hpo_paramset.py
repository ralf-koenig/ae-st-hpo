import math
import random
from pathlib import Path
import pandas as pd

def random_value_log_scale(min_val, max_val):
    """
    Returns a value between min_val and max_val that is evenly
    distributed on the logarithmic scale.
    """
    if 0 < min_val < max_val:
        u = random.random()  # uniform in [0,1]
        log_min = math.log(min_val)
        log_max = math.log(max_val)
        log_val = log_min + u * (log_max - log_min)
        result = math.exp(log_val)
        return result
    else:
        raise ValueError

def get_random_config():
    """
    Get a random configuration for AutoEncodix with some reasonable bounds

    :return:
    config - a config dict
    """
    cfg = dict()
    # Model parameters
    cfg['K_FILTER'] = random.choice([128, 256, 512, 1024, 2048, 4096])  # Input features per data modality
    cfg['N_LAYERS'] = random.choice([2, 3, 4])  # OPTIONAL number of layers for the encoder, respectively decoder (all integer values are possible)
    cfg['ENC_FACTOR'] = random.uniform(1, 4)  # OPTIONAL Factor (float) by which the number of input neurons is divided to get the number of output neurons. Should be larger than 1 for compressing autoencoder.
    cfg['LATENT_DIM_FIXED'] = random.choice([2, 4, 8, 16, 32, 64])  # Latent space dimension
    # Training parameters
    cfg['LR_FIXED'] = random_value_log_scale(1.0e-5, 0.1)  # Learning rate
    cfg['BETA'] = random_value_log_scale(.001, 10)  # weighting of VAE loss for Varix
    cfg['BATCH_SIZE'] = random.choice([32, 64, 128, 256]) # 16 removed - too time-consuming, no value
    cfg['DROP_P'] = random.uniform(0, 0.9)  # We have a small number of samples and should be aggressive with drop out to avoid overfitting
    return cfg

if __name__ == '__main__':
    """
    Create a list of hyperparam value combinations to use for multiple AutoEncodix runs later
    and save them in Parquet format.
    Reason: Syne Tune later needs a common (identical) set of hyperparameter value combinations
    to create a common blackbox from them. 
    """
    SEED_VALUE = 42
    random.seed( SEED_VALUE )  # to get the same configurations each time
    NUMBER_OF_CONFIGS = 5000

    # append first record directly to work around warning from pamdas on concat on empty dataframes
    cfg = pd.DataFrame(get_random_config(), index=[0])
    hyperparam_configs_df = cfg.copy()
    # now append all the rest
    for i in range(NUMBER_OF_CONFIGS-1):
        cfg = pd.DataFrame(get_random_config(), index=[0])
        hyperparam_configs_df = pd.concat([hyperparam_configs_df, cfg], ignore_index=True)

    hyperparam_configs_df.to_parquet(f"hyperparam_configs_{NUMBER_OF_CONFIGS}.parquet")
