"""
Functions for calculations involving inference with a logistic regression classifier.
"""

import numpy as np

type NumpyFloat32Array2D = np.ndarray[tuple[int, int], np.dtype[np.float32]]

def linear_classifier_subscores(coefficients: NumpyFloat32Array2D,
                                values: NumpyFloat32Array2D) ->\
                                tuple[NumpyFloat32Array2D, NumpyFloat32Array2D]:
    """
    Calculate positive and negative linear classifier sub-scores and return them
    separately. By "positive subscore" and "negative subscore" we mean the dot
    product of the positive coefficients and the corresponding input values, and
    the dot product of the negative coefficients and the corresponding input
    values, respectively.

    Parameters
    ----------
    coefficients
        Coefficients of the linear classifier to be multiplied with the input values
    values
        Input numpy array to be converted to the positive and negative linear classi-
        -fier scores. 

    Returns
    -------
    tuple[np.ndarray[tuple[int, int], np.dtype[np.float32], 
          np.ndarray[tuple[int, int], np.dtype[np.float32]]
        Negative and positive subscores calculated using the linear classifier coeffi-
        -cients as numpy arrays. 
    """
    _coeff = coefficients[coefficients.argsort()]
    _values = values[coefficients.argsort(), :]
    res = _coeff[:, np.newaxis] * _values
    return np.sum(res[_coeff < 0.0, :], axis=0), np.sum(res[_coeff >= 0.0, :], axis=0)

def linear_classifier_score(
    coefficients: NumpyFloat32Array2D, values: NumpyFloat32Array2D
) -> float | NumpyFloat32Array2D:
    """
    Calculate the score of a linear classifier by multiplying the coefficients with
    given data. By a "linear classifier" we mean a linear function of the input $x$.
    Given coefficients $\beta$, we calculate the output $y$ as
    
    $$
    y = \beta x
    $$

    Parameters
    ----------
    coefficients
        Coefficients of the linear classifier ($\beta$ in the above equation) to be
        multiplied with the input values
    values
        Input numpy array to be converted to the linear classifier scores. 

    Returns
    -------
    float | np.ndarray[tuple[int, int], np.dtype[np.float32]]
        Value(s) $y$ as the output of the linear transformation as a numpy array. 
    """
    return np.sum(coefficients * values, axis=0)

def antilogit_classifier_score(linear_score: float | NumpyFloat32Array2D,
                               gamma: float = 0.0) -> float | NumpyFloat32Array2D:
    """
    Function to perform anti-logit operation on a linear score. A linear score
    is obtained by a dot product of the classifier coefficients with the input 
    vector (using the "linear classifier" functions). The anti-logit transform
    converts the score from a "linear" scale to a probability scale ([0, 1]).

    $$
    p = \frac{1}{1+e^{-x + \gamma}}
    $$

    Parameters
    ----------
    linear_score
        Data to be converted to a probability score.
    gamma
        Constant value or bias to be added to the linear score before conversion.
        Default is 0.0.

    Returns
    -------
    float | np.ndarray[tuple[int, int], np.dtype[np.float32]]
        Value(s) in the probability scale ([0, 1]) as a numpy array with the same
        shape as the input.   
    """
    return np.exp(gamma + linear_score) / (1 + np.exp(gamma + linear_score))

def z_score(
    x: float | NumpyFloat32Array2D,
    mean: float | NumpyFloat32Array2D, 
    std: float | NumpyFloat32Array2D
) -> float | NumpyFloat32Array2D:
    """
    Given means and standard deviations of the data, converts given data to 
    z-scores. For a measurement $x$, mean $\mu$ and standard deviation $\sigma$,
    a z-score is calculated as
    
    $$
    z = \frac{x - \mu}{\sigma}
    $$

    Parameters
    ----------
    x
        Data to be converted to a z-score.
    mean
        Mean of the distribution of the data.
    std
        Standard deviation of the distribution of the data.

    Returns
    -------
    float | np.ndarray[tuple[int, int], np.dtype[np.float32]]
        A float value or a numpy array representing the z-score.
    """
    return (x - mean) / std