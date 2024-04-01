from setuptools import setup, find_packages

setup(
    name='bayes-opt',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.8.2',
        'numpy==1.24.4',
        'scikit-learn==1.3.0',
        'scipy==1.8.1',
        'streamlit==1.32.2'
    ],
)
