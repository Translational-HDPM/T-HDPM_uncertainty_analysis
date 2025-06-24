"""
Functions for Monte Carlo simulations with a logistic regression classifier.
"""

import numpy as np

type NumpyFloat32Array2D = np.ndarray[tuple[int, int], np.dtype[np.float32]]

def linear_classifier_subscores(coefficients: NumpyFloat32Array2D,
                                samples: NumpyFloat32Array2D) ->\
                                tuple[NumpyFloat32Array2D, NumpyFloat32Array2D]:
    """
    Calculate the positive and negative linear classifier sub-scores and
    return them separately.
    """
    _coeff = coefficients[coefficients.argsort()]
    _samples = samples[coefficients.argsort(), :]
    res = _coeff[:, np.newaxis] * _samples
    return np.sum(res[_coeff < 0.0, :], axis=0), np.sum(res[_coeff >= 0.0, :], axis=0)

def linear_classifier_score(
    coefficients: NumpyFloat32Array2D, values: NumpyFloat32Array2D
) -> float | NumpyFloat32Array2D:
    """
    Calculate the score of a linear classifier by multiplying the coefficients with
    given data.
    """
    return np.sum(coefficients * values, axis=0)

def antilogit_classifier_score(linear_score: float | NumpyFloat32Array2D,
                               gamma: float = 0.0) -> float | NumpyFloat32Array2D:
    """
    Function to perform anti-logit operation on a linear score
    """
    return np.exp(gamma + linear_score) / (1 + np.exp(gamma + linear_score))

def z_score(
    x: float | NumpyFloat32Array2D,
    mean: float | NumpyFloat32Array2D, 
    std: float | NumpyFloat32Array2D
) -> float | NumpyFloat32Array2D:
    """
    Given means and standard deviations of the data, converts given data to 
    z-scores.
    """
    return (x - mean) / std