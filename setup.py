from setuptools import find_packages, setup

setup(
    name='kfsims',
    version='1.0.0', 
    packages=find_packages(),
    description="Library implementing Distributed VBAKF",
    python_requires=">=3.6",
    licence="MIT",
    url='https://github.com/hnykda/kfsmis',
    install_requires=[
        "filterpy==1.2.1",
        "matplotlib==2.2.2",
        "networkx==2.1",
        "numpy==1.14.2",
        "pandas==0.22.0",
        "scipy==1.0.0",
        "seaborn==0.8.1",
    ],
)
