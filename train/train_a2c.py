import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs import ACConfig, build_default_configs
from runners.actor_critic.actor_critic import main


if __name__ == "__main__":
    args = build_default_configs(ACConfig)
    main(args)
