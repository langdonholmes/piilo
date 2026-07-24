import yaml

from piilo.resources import package_path

CONFIG_FILE_PATH = package_path("configs", "kaggle_third.yaml")

with open(CONFIG_FILE_PATH, "r") as f:
    piilo_config = yaml.safe_load(f)
