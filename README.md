# Shallow Universal Polynomial Networks

[![Python Tests](https://github.com/sandialabs/SUPN/actions/workflows/python-tests.yml/badge.svg)](https://github.com/sandialabs/SUPN/actions/workflows/python-tests.yml)
[![Coverage](https://github.com/sandialabs/SUPN/blob/gh-pages/badges/coverage.svg)](https://github.com/sandialabs/SUPN/actions)
[![DOI](https://img.shields.io/badge/any_test-2511.21414-blue?style=flat&label=arXiv&link=https%3A%2F%2Farxiv.org%2Fabs%2F2511.21414)](https://arxiv.org/abs/2511.21414)

SUPNs (Shallow Universal Polynomial Networks) are parsimonious, yet highly expressive, surrogate models of the form

```math
\hat{f}(x) = \sum_{n=1}^N w_n \tanh\left( \sum_{m=0}^M a_{n,m} \, T_m(x) \right) \, ,
```

where $`T_m`$ is the Chebyshev polynomial of degree $m$. SUPNs have been shown to achieve equal or better accuracy to DNNs, KANs, and even polynomials (on nonsmooth problems) with a similar parameter count \[1\]. They also come equipped with convergence theory that goes beyond the usual universal approximation theorems. Single-neuron SUPNs have accuracy bounded by the best polynomial approximation of a given degree $M$ \[1\]. Multi-neuron SUPNs are capable of approximating a spline interpolant with accuracy explicitly in terms of basis size, spline smoothness, and $M$ \[2\].

## Installation
```bash
git clone <this-repo>
cd SUPN
pip install -e .
```

## Usage
See the [`tutorials`](./tutorials/) folder.

## Dependencies

- PyTorch

# Acknowledgements

Development of this software was funded by Laboratory-Directed Research and Development (LDRD) program at Sandia National Laboratories.

## Citations

If you use this code in your research, please use the following citations:

1. Z. Morrow, M. Penwarden, B. Chen, A. Javeed, A. Narayan, and J. D. Jakeman. "SUPN: Shallow Universal Polynomial Networks." arXiv preprint [arXiv:2511.21414](https://arxiv.org/abs/2511.21414) (2025).

2. Z. Morrow, Y. Fu, P. Roy, and M. Penwarden. "Physics-Informed Shallow Universal Polynomial Networks for High-Frequency and High-Order Systems." In preparation (2026).

BibTeX:

```bibtex
@misc{morrow2025supn,
    title = {SUPN: Shallow Universal Polynomial Networks},
    author = {Zachary Morrow and Michael Penwarden and Brian Chen
              and Aurya Javeed and Akil Narayan and John D. Jakeman},
    archivePrefix = {arXiv},
    eprint = {2511.21414},
    year = {2025},
}
```

