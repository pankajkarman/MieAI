# MieAI

Aerosol radiative modeling (Mie) emulation using Artificial Intelligence

This repo uses Core Shell model for coated spheres and contains some functions from [PyMieScat](https://pymiescatt.readthedocs.io/en/latest/) which are modified to match the MATLAB code from Tami Bond.

# Analysis codes

1. [Mie calculation for core-shell configuration](Mie_calculation.ipynb)
2. [Compilation of Mie results in a single file](read_mie_data_10m.ipynb)
3. [MLP training without quantile transform: x1](mlp_training_double_x1_10m.ipynb)
4. [MLP training without quantile transform: x2](mlp_training_double_x2_10m.ipynb)

5. [Quantile Transformation: Figure3 in paper](quantile_transform.ipynb)

6. [Hyperparamter optimisation of MLP architecture: Part1](mlp_hyper1.ipynb)
7. [Hyperparamter optimisation of MLP architecture: Part2](mlp_hyper2.ipynb)
8. [Hyperparamter optimisation Table](read_hyper.ipynb)

9. [MieAI Training](MieAI_training.ipynb)
10. [MieAI performance: Figure4 in paper](MieAI_performance.ipynb)
11. [AOD calculation using MieAI](AOD.ipynb)

# utility codes

1. [Codes translated from MATLAB2python](mei.py)
2. [Functions for preprocessing, mie run and MLP](mie_icon_art.py)
3. [AOD calculation using ICON-ART data](aop.py)


# Use cases
2. [Code for Biomass Burning Event](wildfire.ipynb)
2. [Code for Volcanic Eruption Event](volcano.ipynb)
2. [Code for Dust Event](dust.ipynb)


# References

1. [Papers](paper.md)
2. [Packages](software.md)


