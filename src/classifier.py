"""
Functions for Monte Carlo simulations with a logistic regression classifier.
"""

import numpy as np


def linear_classifier_subscores(coefficients: np.ndarray[np.float32],
                                samples: np.ndarray[np.float32]) ->\
                                tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
    """
    Calculate the positive and negative linear classifier sub-scores and
    return them separately.
    """
    _coeff = coefficients[coefficients.argsort()]
    _samples = samples[coefficients.argsort(), :]
    res = _coeff[:, np.newaxis] * _samples
    return np.sum(res[_coeff < 0.0, :], axis=0), np.sum(res[_coeff >= 0.0, :], axis=0)

def linear_classifier_score(
    coefficients: np.ndarray[np.float32], values: np.ndarray[np.float32]
) -> float | np.ndarray[np.float32]:
    """
    Calculate the score of a linear classifier by multiplying the coefficients with
    given data.
    """
    return np.sum(coefficients * values, axis=0)

def antilogit_classifier_score(linear_score: float | np.ndarray[np.float32],
                               gamma: float = 0.0) -> float | np.ndarray[np.float32]:
    """
    Function to perform anti-logit operation on a linear score
    """
    return np.exp(gamma + linear_score) / (1 + np.exp(gamma + linear_score))

def z_score(
    x: float | np.ndarray[np.float32],
    mean: float | np.ndarray[np.float32], 
    std: float | np.ndarray[np.float32]
) -> float | np.ndarray[np.float32]:
    """
    Given means and standard deviations of the data, converts given data to 
    z-scores.
    """
    return (x - mean) / std