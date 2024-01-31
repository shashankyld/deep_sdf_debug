#!/bin/bash -e

conda env create -f environment.yml --force

conda_base=$(conda info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate deep_sdf_debug
