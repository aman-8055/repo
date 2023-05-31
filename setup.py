from setuptools import setup, find_packages

setup(
    name='pegasus_summarizer',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'torch+cpu',
        'transformers'
    ],
    author='Your Name',
    description='Streamlit application for paragraph summarization using Pegasus',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Web Environment',
        'Framework :: Streamlit',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
