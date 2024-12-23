"""Logistic regression classifier functions."""

import numpy as np
from typing import Sequence

np.random.seed(46215423)


def sample(means: Sequence[float], std: Sequence[float], coefficients: Sequence[float]):
    """Sampling function performing the Monte Carlo simulations"""
    return np.sum(
        np.multiply(
            coefficients, np.random.normal(means, std, size=(1, len(coefficients)))
        )
    )


def linear_classifier_score(
    coefficients: Sequence[float], col: Sequence[float]
) -> float:
    """This score is the classifer linear score we want to compare with the simulated scores."""
    return np.sum(coefficients * col, axis=0)


def antilogit_classifier_score(linear_score: float, gamma: float = 0.0):
    """Function to perform anti-logit operation on the linear score"""
    return np.exp(gamma + linear_score) / (1 + np.exp(gamma + linear_score))


def sample_single_patient(
    col: Sequence[float],
    coefficients: Sequence[float],
    num_runs: int = 100,
    percent: float = 100,
) -> Sequence[float]:
    """Function to calculate the classifier score for each simulation of "num_runs" simulations, corresponding to each subject."""
    if not 0.0 <= percent <= 100.0:
        raise RuntimeError("Percent out of bounds.")
    return np.asarray(
        [
            antilogit_classifier_score(
                sample(
                    col, np.abs([percent / 100.0 * val for val in col]), coefficients
                )
            )
            for _ in range(num_runs)
        ],
        dtype=np.float32,
    )


def z_score(
    x: Sequence[float], mean: Sequence[float], std: Sequence[float]
) -> Sequence[float]:
    """Function whose input is TPM and output the corresponding Z-score."""
    return (x - mean) / std
