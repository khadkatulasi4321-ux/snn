from setuptools import setup, find_packages
import os

HERE = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="snn",
    version="0.1.3",
    author="samsit-phew",
    description="Simple Neural Network — NumPy deep learning library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samsit-phew/snn",
    license="MIT",
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "docs": [
            "sphinx>=7.0",
            "furo>=2023.1",
            "myst-parser>=2.0",
            "sphinx-autodoc-typehints>=1.23",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "matplotlib>=3.7",
            "scipy>=1.10",
        ],
        "all": [
            "sphinx>=7.0",
            "furo>=2023.1",
            "myst-parser>=2.0",
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "matplotlib>=3.7",
            "scipy>=1.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "snn-docs=snn.docs.serve_docs:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords=[
        "deep-learning",
        "numpy",
        "neural-networks",
        "from-scratch",
        "backpropagation",
    ],
)
