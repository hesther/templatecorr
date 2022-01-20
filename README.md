templatecorr
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/hesther/templatecorr/workflows/CI/badge.svg)](https://github.com/hesther/templatecorr/actions?query=workflow%3ACI)


A Python package to hierarchically correct a list of reaction templates.
See our publication [On the influence of template size, canonicalization and exclusivity for retrosynthesis and reaction prediction applications](https://doi.org/10.1021/acs.jcim.1c01192) for further information and benchmarks.

### Installation

Download templatecorr from Github:

```
git clone https://github.com/hesther/templatecorr.git
cd templatecorr
```

Set up a conda environment (or install the packages in `environment.yml` in any other way convenient to you):

```
conda env create -f environment.yml
conda activate templatecorr
```

This environment uses the RDChiral C++ package to extract and canonicalize templates. If you instead want to use the older RDChiral Python package for backwards compatibility reasons, use the file environment_scripts.yml instead of environment.yml. Also, RDChiral C++ is currently not available on Windows, in this case, also use environment_scripts.yml.

Install the template corr package:

```
pip install -e .
```

### Extract and correct templates from a file

To extract hierarchically corrected templates from a set of reactions, use `correct.py`. Prepare a `csv`, `pkl/pickle` of a pandas dataframe, `json/json.gz` or `hdf5` of a pandas dataframe file of reactions. The file must at least contain one column with atom-mapped reaction SMILES. The column can have any name, default `rxn_smiles`, which can be specified via `--reaction_column`. Other information (other columns) in the files will be conserved. In the following we will use `data/uspto_50k.csv` (extract the archive `data.tar.gz` to access it). You can use the default arguments:

```
python correct.py --path data/uspto_50k
```

where `--path` specifies the path to the reaction file without file ending. This will create the file data/uspto_50k_corrected.csv, which now contains an additional column `template`, holding the extracted, canonicalized and corrected template. The above command is the same as

```
python correct.py --path data/uspto_50k --reaction_column rxn_smiles --name template --nproc 20 --drop_extra_cols --data_format csv
```

where `--reaction_column rxn_smiles` specifies the name of the column containing reaction SMILES, `--name template` sets the name of the column for the extracted templates in the output file (here to "template"), `--nproc 20` parallelizes the program over 20 processes, `--drop_extra_cols` causes additional helper columns during extraction (canonical reactant SMILES, templates at radius 0 and 1) to be dropped before saving the dataframe to file, and `--data_format csv` specifies the input format of the data, as well as the output format.

### Use to retrain a template relevance model

If you want to use the template correction code together with the [template-relevance](https://gitlab.com/mefortunato/template-relevance) GitLab repository, there is a simple drop-in replacement: In your workflow, instead of using bin/process.py from the template-relevance repository, use temprel_scripts/process.py (same usage, same arguments).


### Reproduce our study

If you want to reproduce the results of the publication [On the influence of template size, canonicalization and exclusivity for retrosynthesis and reaction prediction applications](https://doi.org/10.1021/acs.jcim.1c01192), you need to create another conda environment (since the newest rdchiral version (C++) used above automatically canonicalizes templates). We will use both environments in the following.

```
conda deactivate
conda env create -f environment_scripts.yml
conda activate templatecorr_nocan
pip install -e .
```

Extract the archived file `data.tar.gz` if you have not already done so, go to the `scripts` folder  and run the scripts in consecutive order (from 01 to 05). To reproduce the exact results from the manuscript, run script 01, 03, 04 and 05 with the templatecorr_nocan environment, and script 02 with the templatecorr environment. Since the canonicalization functionality is now per default used in the C++ rdchiral package, non-canonical templates can otherwise not be obtained easily.

### AiZynthFinder models

AiZynthFinder template and policy model files are available in the folder `aizynthfinder_models` for the canonical-corrected template sets of this study (USPTO-50k and USPTO-460k).

### Contact

For questions, feedback, concerns or wishes, contact Esther at eheid@mit.edu.

### Copyright

Copyright (c) 2021, Esther Heid


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.
