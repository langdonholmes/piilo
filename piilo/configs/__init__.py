import os
import pkg_resources
import yaml

CONFIG_FILE_PATH = pkg_resources.resource_filename('piilo', os.path.join("configs", "kaggle_third.yaml"))

with open(CONFIG_FILE_PATH, 'r') as f:
    piilo_config = yaml.safe_load(f)