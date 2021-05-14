#!/usr/bin/env bash

conda env create -f environment.yaml \
&& source $HOME/anaconda3/etc/profile.d/conda.sh \
&& conda activate argoverse \
&& poetry install
