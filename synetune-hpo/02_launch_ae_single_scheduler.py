"""
This example shows how to run CQR on benchmarks from AutoEncodix by Joas et al.
We use a XGBoost surrogate model to predict the performance of unobserved hyperparameter configurations.
"""
import logging
from syne_tune.blackbox_repository import (
    load_blackbox,
    BlackboxRepositoryBackend,
)

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import CQR, TPE, BORE, REA, RandomSearch
from syne_tune import StoppingCriterion, Tuner
from syne_tune.config_space import choice

from syne_tune.experiments import load_experiment
import matplotlib.pyplot as plt

BLACKBOX_NAME = "autoencodix_vanillix"
# BLACKBOX_NAME = "autoencodix_varix"

# DATASET_NAME = "tcga_rna"
# DATASET_NAME = "tcga_meth"
# DATASET_NAME = "tcga_dna"
# DATASET_NAME = "schc_rna"
DATASET_NAME = "schc_meth"

# Target metric
# metric = "valid_r2"
metric = "WEIGHTED_AVG_AUC_DOWNSTREAM_PERFORMANCE"

# maximize AUC (downstream task performance)
# alternatively: maximize validation_r2
DO_MINIMIZE=False

def simulate_benchmark(blackbox, trial_backend, metric):
    max_resource_attr = "epoch"

    my_config_space=blackbox.configuration_space_with_max_resource_attr(
            max_resource_attr
        )
    # Optional: Customize config space to desired target properties
    # my_config_space['LATENT_DIM_FIXED'] = choice([4])
    # my_config_space['BATCH_SIZE'] = choice([256])

    scheduler = CQR( # Single Objective Schedulers
                    # CQR  - (Single-fidelity) Conformal Quantile Regression (Salinas, 2023)
                    # BORE - Bayesian Optimization by Density-Ratio Estimation (2021)
                    # REA  - Regularized Evolution (Real, 2019)
                    # TPE  - Tree-Parzen Estimator (Bergstra, 2011)
        # scheduler = RandomSearch( # SingleFidelity
        config_space=my_config_space,
        # max_resource_attr=max_resource_attr,
        # time_attr=blackbox.fidelity_name(),
        metric=metric, # all Single Objective searchers get a single of target variables
        # metrics=[metric], # all Multi Objective searchers ge  t a list of target variables
        do_minimize=DO_MINIMIZE,
        random_seed=31415927,
      )

    stop_criterion = StoppingCriterion(max_wallclock_time=3600 * 4) # simulated seconds

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        callbacks=[SimulatorCallback()],
    )
    tuner.run()

    # Display plots on progress
    tuning_experiment = load_experiment(tuner.name)
    print(f"\n=================================\nOverall best result found (single scheduler):")
    print("\n".join(f"{k}: {v}" for k, v in tuning_experiment.best_config().items()))

    tuning_experiment.plot()
    tuning_experiment.plot_trials_over_time(metric_to_plot=metric)
    plt.show()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    n_workers = 6

    blackbox = load_blackbox(BLACKBOX_NAME)[DATASET_NAME]
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=BLACKBOX_NAME,
        dataset=DATASET_NAME,
        elapsed_time_attr="training_runtime",
        surrogate="XGBRegressor",
    )
    simulate_benchmark(blackbox=blackbox, trial_backend=trial_backend, metric=metric)
