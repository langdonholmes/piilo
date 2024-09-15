import toml
from setuptools import setup, find_packages
from pathlib import Path
import argparse

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

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

setup(
    name='piilo',
    version='0.1.7',
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