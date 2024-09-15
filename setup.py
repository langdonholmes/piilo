import toml
from setuptools import setup, find_packages

README_FILE = "README.md"
POETRY_FILE = "poetry.lock"
VERSION = "0.1.8"

with open(README_FILE, 'r') as f:
    long_description = f.read()

def extract_dependencies_from_poetry_lock():
    with open(POETRY_FILE, 'r') as lock_file:
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

setup(
    name='piilo',
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "piilo.configs": ["*.yaml"],
        "piilo.data": ["*.parquet"],
        "piilo.models": ["*.pkl", "*.json"],
    },
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type='text/markdown',
    entry_points={
        "console_scripts": [
            "obfuscate=piilo:anonymize_batch_cli",
        ],
    },
)