from setuptools import find_packages, setup

setup(
    name='kfsims',
    version='0.1.0',  # always SemVer2
    packages=find_packages(),
    description="informative description",
    python_requires=">=3.6",
    licence="MIT",
    url='https://github.com/GlobalWebIndex/appname',
    install_requires=[
        "networkx",
        "matplotlib",
        "seaborn",
        "pandas",
        "numpy",
        "scipy",
        "filterpy",
    ],
)
