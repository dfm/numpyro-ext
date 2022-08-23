from setuptools import find_packages, setup

setup(
    name="numpyro-ext",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
