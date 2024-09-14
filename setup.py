import toml
from setuptools import setup, find_packages

def extract_dependencies_from_poetry_lock():
    with open('poetry.lock', 'r') as lock_file:
        lock_data = toml.load(lock_file)
    
    dependencies = []

    for package in lock_data['package']:
        name = package['name']
        version = package['version']
        # Exclude spacy model
        if "en_core_web_sm" not in name:
            dependencies.append(f"{name}=={version}")
    
    return dependencies

install_requires = extract_dependencies_from_poetry_lock()

# Monkey patch before including xgboost in poetry.lock:
install_requires += ['xgboost==2.0.3', 'scikit-learn==1.5.1']

setup(
    name='piilo',
    version='0.1.6',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "piilo.configs": ["*.yaml"],
        "piilo.data": ["*.parquet"],
        "piilo.models": ["*.pkl", "*.json"],
    },
    install_requires=install_requires,
)