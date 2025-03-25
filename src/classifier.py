"""
Functions for Monte Carlo simulations with a logistic regression classifier.
"""

import numpy as np
from typing import Sequence

def linear_classifier_subscores(coefficients: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the positive and negative linear classifier sub-scores and return them separately."""
    _coeff, _samples = coefficients[coefficients.argsort()], samples[coefficients.argsort(), :]
    res = _coeff[:, np.newaxis] * _samples
    return np.sum(res[_coeff < 0.0, :], axis=0), np.sum(res[_coeff >= 0.0, :], axis=0)

def linear_classifier_score(
    coefficients: np.ndarray, col: np.ndarray
) -> float:
    """This score is the classifer linear score we want to compare with the simulated scores."""
    return np.sum(coefficients * col, axis=0)


def antilogit_classifier_score(linear_score: float | np.ndarray, gamma: float = 0.0) -> float | np.ndarray:
    """Function to perform anti-logit operation on the linear score"""
    return np.exp(gamma + linear_score) / (1 + np.exp(gamma + linear_score))


def z_score(
    x: float | np.ndarray, mean: float | np.ndarray, std: float | np.ndarray
) -> float | np.ndarray:
    """Function whose input is TPM and output the corresponding Z-score."""
    return (x - mean) / std

def sample(means: Sequence[float], std: Sequence[float], coefficients: Sequence[float]) -> float:
    """Sampling function performing the Monte Carlo simulations"""
    return np.sum(
        np.multiply(
            coefficients, np.random.normal(means, std, size=(1, len(coefficients)))
        )
    )

def sample_single_patient(
    col: Sequence[float],
    coefficients: Sequence[float],
    num_runs: int = 100,
    percent: float = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Function to calculate the classifier score for each simulation of "num_runs" simulations, corresponding to each subject."""
    if not 0.0 <= percent <= 100.0:
        raise RuntimeError("Percent out of bounds.")
    lin_scores = np.asarray([sample(
                    col, np.abs([percent / 100.0 * val for val in col]), coefficients
                ) for _ in range(num_runs)], dtype=np.float32)
    return lin_scores, antilogit_classifier_score(lin_scores)