from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.blackbox_surrogate import add_surrogate

from xgboost import XGBRegressor

BLACKBOX = "autoencodix_varix"
DATA_SCENARIO_TASK = "schc_rna"

blackbox = load_blackbox(BLACKBOX)[DATA_SCENARIO_TASK]
# blackbox = load_blackbox("autoencodix_varix")["tcga_meth"]

print("Configuration Space:")
print(*[f"{k}: {v}" for k, v in blackbox.configuration_space.items()], sep="\n")

# Create a surrogate model from data
surrogate_blackbox = add_surrogate(
    blackbox,
    # we need "predict_curves=False", so that the function returns a Dict of all target metrics and values
    predict_curves=False,
    # default surrogate_blackbox (when 'surrogate' parameter is None: (kNearestNeighbours(1)) is deterministic on the supplied data
    #         surrogate_blackbox is also deterministic for XGBoostRegressor() with default hyperparameters
    # surrogate=XGBRegressor()
)

print("\nOne random sample configuration from this space and the predicted result on a trained surrogate model:")
config = {k: v.sample() for k, v in surrogate_blackbox.configuration_space.items()}

config = {'K_FILTER': 256, 'N_LAYERS': 4, 'ENC_FACTOR': 2.4976151430258473, 'LATENT_DIM_FIXED': 16, 'BATCH_SIZE': 128,
          'LR_FIXED': 0.034402955998218514, 'DROP_P': 0.2632518199364008}
# only set BETA for a Variational AutoEncoder ("varix") as it is not in the config_space of Vanillix
if BLACKBOX=="autoencodix_varix":
    config["BETA"] = 0.1
print(config)

# surrogate_result is deterministic for default surrogate k-Nearest-Neighbour(1)
# surrogate_result is deterministic for XGBoostRegressor with default hyperparameters
surrogate_result = surrogate_blackbox(config, fidelity={'epoch': 0})
print(surrogate_result)
