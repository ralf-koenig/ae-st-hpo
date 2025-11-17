from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.blackbox_surrogate import add_surrogate

from xgboost import XGBRegressor

BLACKBOX = "fcnet"
# DATA_SCENARIO_TASK = ""

blackbox = load_blackbox(BLACKBOX)

print(blackbox)