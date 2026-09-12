"""Fruit fly connectome as a spiking brain, wired to the flybody MuJoCo body."""

import warnings

__version__ = "0.1.0"

# torch prints this once per process for every sparse CSR tensor we build
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")
