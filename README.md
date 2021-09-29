templatecorr
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/hesther/templatecorr/workflows/CI/badge.svg)](https://github.com/hesther/templatecorr/actions?query=workflow%3ACI)


A Python package to hierarchically correct a list of reaction templates.

### Installation

Download templatecorr from Github:

```
git clone https://github.com/hesther/templatecorr.git
cd templatecorr
```

Set up a conda environment (or install the packages rdkit, pandas and numpy in any other way convenient to you). If you just want to use the correction code, use:

```
conda env create -f environment.yml
conda activate templatecorr
```

If you want to use the scripts in the `scripts` folder, you will need a few additional packages (joblib, tensorflow, sklearn). In this case, instead type:

```
conda env create -f environment_scripts.yml
conda activate templatecorr
```

In both cases, install the template corr package:

```
pip install -e .
```

Install rdchiral from the fork [https://github.com/hesther/rdchiral](https://github.com/hesther/rdchiral) and install via

```
git clone https://github.com/hesther/rdchiral.git
cd rdchiral
pip install -e .
```

Do not install rdchiral via the Pypi package (which does not include functionality to extract templates at different radii). You are now ready to extract and correct templates hierarchically. For usage examples, refer to the `scripts` folder.

### Reproduce our study

If you want to use the canonicalization functionality, you must install the RDChiral C++ drop-in. It is currently only available for Unix. If you want to have both RDChiral versions available on your system, consider setting up a separate conda environment. The following command installs it in a separate conda environment. If you have another active conda environment, first deactivate it via `conda deactivate`.

```
conda create -n rdchiral_cpp
conda activate rdchiral_cpp
conda install -c conda-forge -c ljn917 rdchiral_cpp
```

and then again install the templatecorr package (`pip install -e .` in the templatecorr folder). Installing RDChiral C++ is optional, the hierarchical correction code does not rely on the canonicalization functionality. If the RDChiral C++ drop-in is not installed, the canonicalization function simply returns the unchanged template.

To reproduce results from the publication [On the influence of template size, canonicalization and exclusivity for retrosynthesis and reaction prediction applications](https://github.com/hesther/templatecorr), extract the archived file `data.tar.gz`, go to the `scripts` folder  and run the scripts in consecutive order (from 01 to 05). To reproduce the exact results from the manuscript, run `01_data_preparation.py` with the Python rdchiral package, and `02_template_correction.py` with the C++ rdchiral package. Since the canonicalization functionality is now per default used in the C++ rdchiral package, non-canonical templates can otherwise not be obtained easily.


### Copyright

Copyright (c) 2021, Esther Heid


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.
