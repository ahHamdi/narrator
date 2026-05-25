from setuptools import setup, find_packages

setup(
    name="narrativebench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "sentence-transformers>=2.7.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.0",
    ],
    python_requires=">=3.10",
)
