"""
Synthetic data generation suite for the Loan Performance Intelligence Engine.

Five phases:
    1. Static attribute space      -- Gaussian copula over credit/LTV/DTI
    2. Longitudinal simulation      -- risk-scaled discrete-time Markov chain
    3. Target engineering           -- vectorised forward windows with censoring
    4. Anomaly and conflict injection
    5. Time-aware partitioning and supporting artifacts
"""

from .config import GenerationConfig
from .pipeline import GenerationResult, SyntheticDataPipeline

__all__ = ["GenerationConfig", "GenerationResult", "SyntheticDataPipeline"]
