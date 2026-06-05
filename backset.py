"""
setup.py for snn — Simple Neural Network.

A pure-NumPy deep learning library exposing a Keras-like API.

    pip install .            # basic install (numpy only)
    pip install .[docs]      # + Sphinx for building documentation
    pip install .[dev]       # + pytest + matplotlib for development
    pip install -e .         # editable / development install
"""

import os
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))

# Long description from README.md
_readme = os.path.join(HERE, "README.md")
if os.path.exists(_readme):
    with open(_readme, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = (
        "snn — A neural network / deep learning library built purely on NumPy. "
        "No TensorFlow, no PyTorch — every forward pass, backward pass, and "
        "weight update is hand-derived and fully vectorised."
    )

setup(
    # ── Identity ──────────────────────────────────────────────────────────────
    name="snn",
    version="0.1.1",
    author="samsit-phew",
    author_email="",
    description=(
        "Simple Neural Network — a pure-NumPy deep learning library "
        "with a Keras-like API"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samsit-phew/snn",
    license="MIT",
    # ── Packages ──────────────────────────────────────────────────────────────
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    # ── Package data (non-.py files shipped inside the package) ───────────────
    package_data={
        "snn": [
            # Bundled Sphinx docs source so `python -m snn.docs.serve_docs`
            "docs/**/*"
            # works after a regular (non-editable) pip install.
        ],
    },
    include_package_data=True,
    # ── Dependencies ──────────────────────────────────────────────────────────
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        # Documentation extras — auto-installed by `python -m snn.docs.serve_docs`
        "docs": [
            "sphinx>=7.0",
            "furo>=2023.1",
            "myst-parser>=2.0",
            "sphinx-autodoc-typehints>=1.23",
        ],
        # Development extras
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "matplotlib>=3.7",
            "scipy>=1.10",
        ],
        # Everything
        "all": [
            "sphinx>=7.0",
            "furo>=2023.1",
            "myst-parser>=2.0",
            "sphinx-autodoc-typehints>=1.23",
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "matplotlib>=3.7",
        ],
    },
    # ── Console scripts ───────────────────────────────────────────────────────
    entry_points={
        "console_scripts": [
            # After `pip install snn-numpy`, run `snn-docs` from any terminal.
            "snn-docs=snn.docs.serve_docs:main",
        ],
    },
    # ── PyPI classifiers ──────────────────────────────────────────────────────
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    keywords=[
        "deep-learning",
        "neural-networks",
        "machine-learning",
        "numpy",
        "from-scratch",
        "education",
        "backpropagation",
    ],
)
