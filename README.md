# SIR-Model Project (Python + Quarto)

This repository contains the code and report for the spatial SIR (PDE) model project.

## Prerequisites
- Python 3.10+ (recommended)
- Quarto installed (needed to render the `.qmd` report)

## Create and install the virtual environment

From the project root (where `requirements.txt` is located):

### 1) Create the environment
`python -m venv sir-model`

### 2) Activate the environment

`.\sir-model\Scripts\Activate.ps1`

### 3) Activate the environment
`python -m pip install --upgrade pip`

`pip install -r requirements.txt`

### 4) Register the environment as a Jupyter kernel
`python -m ipykernel install --user --name sir-model --display-name "Python (sir-model)"`


Reference for Covid Data
https://www.idd.bag.admin.ch/diseases/covid/overview 

Covid-19 SIR-model US:
https://pmc.ncbi.nlm.nih.gov/articles/PMC8993010/#sec002 
