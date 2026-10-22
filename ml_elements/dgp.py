"""
dgp.py
------
Former home of the populations. ``GaussianBinaryDGP``, ``RealDataDGP`` and
``ShiftedDGP`` are aliases for ``GaussianPopulation``, ``ResampledPopulation``
and ``ShiftedPopulation`` in ``population.py``.
"""

from .population import GaussianPopulation, ResampledPopulation, ShiftedPopulation

GaussianBinaryDGP = GaussianPopulation
RealDataDGP = ResampledPopulation
ShiftedDGP = ShiftedPopulation

__all__ = ["GaussianBinaryDGP", "RealDataDGP", "ShiftedDGP"]
