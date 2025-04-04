from setuptools import setup, find_packages

setup(
    name="cell-curvature-analyzer",
    version="1.0.0",
    description="GUI application for analyzing cell-membrane curvature and PIEZO1 protein locations",
    author="Cell Curvature Analysis Team",
    packages=find_packages(),
    package_data={
        "cell_curvature_analyzer": ["icons/*.png"],
    },
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "scikit-image>=0.17.0",
        "pandas>=1.1.0",
        "PyQt5>=5.15.0",
        "tifffile>=2020.9.3",
    ],
    entry_points={
        "console_scripts": [
            "cell-curvature-analyzer=cell_curvature_analyzer.main:main",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
