# nystrom-mmd

This repository contains the code to reproduce the results presented in:

**A. Chatalic, M. Letizia, N. Schreuder, L. Rosasco**  
*A Scalable Nyström-Based Kernel Two-Sample Test with Permutations*  
[arXiv:2502.13570](https://arxiv.org/abs/2502.13570)

To use our Nyström-based MMD permutation test on your data, we reccommend referring to [this repository](https://github.com/mletizia/Nystrom-MMD-test).

---

## Reproducing Results

- **Experiments**: Run the `experiments_*.py` scripts to reproduce the numerical studies presented in the paper.  

For example, to run the Higgs experiments with the parameters used in the paper

python experiments_higgs.py --output_folder results/

- **Analysis & Plots**: Use the Jupyter notebooks in the `analysis/` directory to generate figures and further analyses.

---

## Status
This codebase is currently under final revision.

