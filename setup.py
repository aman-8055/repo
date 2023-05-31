from setuptools import setup, find_packages

setup(
    name='app.py',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'sumy',
        'streamlit',
        'torch==1.9.0',
'transformers==4.11.2',
    ],
)
