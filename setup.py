from setuptools import setup, find_packages

setup(
    name='piilo',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        nameparser,
        presidio_analyzer
    ],
)