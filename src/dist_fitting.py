"""
Functions to test fit of data to various probability distributions.
"""
from typing import Sequence

import pandas as pd
import numpy as np
import scipy.stats as st
import statsmodels.discrete.count_model as cm
import statsmodels.discrete.discrete_model as smd


def test_gamma_fit(data: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Performs Kolmogorov-Smirnov (KS) Test on the fit of the gamma distribution on given data.

    Parameters
    ----------
    data
        Numpy array of data to test fit against.

    Returns
    -------
    tuple[float, float, float, float, float]
        KS Statistic, p-value and parameters from fitting the data to a gamma distribution.
    """
    fit_alpha, fit_loc, fit_beta = st.gamma.fit(data+1, floc=0)
    ks_stat, p_value = st.kstest(data+1, "gamma", args=(fit_alpha, fit_loc, fit_beta))
    return ks_stat, p_value, fit_alpha, fit_loc, fit_beta

def test_lognormal_fit_ks(data: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Performs Kolmogorov-Smirnov (KS) Test on the fit of the log normal distribution on given data.

    Parameters
    ----------
    data
        Numpy array of data to test fit against.

    Returns
    -------
    tuple[float, float, float, float, float]
        KS Statistic, p-value and parameters from fitting the data to a lognormal distribution.
    """
    if np.min(data) == 0:
        data += 1
    fit_shape, fit_loc, fit_scale = st.lognorm.fit(data, floc=0)
    ks_stat, p_value = st.kstest(data, 'lognorm', args=(fit_shape, fit_loc, fit_scale))
    return ks_stat, p_value, fit_shape, fit_loc, fit_scale

def test_lognormal_fit_sw(data: np.ndarray) -> tuple[float, float]:
    """
    Performs Shapiro-Wilk Test on the fit of the log normal distribution on given data.

    Parameters
    ----------
    data
        Numpy array of data to test fit against.

    Returns
    -------
    tuple[float, float]
        SW Statistic and p-value.
    """
    if np.min(data) == 0.0:
        data += 1
    log_data = np.log2(data)
    stat, p_value = st.shapiro(log_data)
    return stat, p_value

def test_negative_binomial_fit(data: Sequence[float]) -> tuple[float, float, float, float, float]:
    """
    Determines fit of a negative binomial distribution against given data.

    Parameters
    ----------
    data
        Data to test fit against.

    Returns
    -------
    tuple[float, float, float, float, float]
        AIC value, KS Statistic and p-value and parameters of the distribution fit.
    """
    data = pd.Series(np.asarray(data, dtype=np.int64))
    nb_model = smd.NegativeBinomial(data, np.ones_like(data)[:, np.newaxis])
    nb_results = nb_model.fit(disp=0)
    alpha, lambda_nb = nb_results.params[1], np.exp(nb_results.params[0])
    r = 1.0 / alpha
    p = r / (r + lambda_nb)
    def cdf_nb(x: Sequence[float]) -> np.ndarray:
        return st.nbinom.cdf(x, r, p)
    ks_stat, p_value = st.kstest(data, cdf_nb)
    return nb_results.aic, ks_stat, p_value, r, p

def test_poisson_fit(data: Sequence[float]) -> tuple[float, float, float, float]:
    """
    Determines fit of a Poisson distribution and returns the Akaike Information Criterion, 
    Kolmogorov-Smirnov (KS) test statistic, p-value and parameters of the fit distribution.

    Parameters
    ----------
    data
        Data to test fit against.

    Returns
    -------
    tuple[float, float, float, float]
        AIC value, KS Statistic and p-value and parameters of the distribution fit.
    """
    data = pd.Series(np.asarray(data, dtype=np.int64))
    poisson_model = smd.Poisson(data, np.ones_like(data)[:, np.newaxis])
    poisson_results = poisson_model.fit(disp=0)
    lambda_p = np.exp(poisson_results.params[0])
    def cdf_p(x: Sequence[float]) -> np.ndarray:
        return st.poisson.cdf(x, mu=lambda_p)
    ks_stat, p_value = st.kstest(data, cdf_p)
    return poisson_results.aic, ks_stat, p_value, lambda_p

def test_zero_inflated_poisson_fit(data: Sequence[float]) \
            -> tuple[float, float, float, float, float]:
    """
    Determines fit of a zero-inflated Poisson distribution and returns the Akaike 
    Information Criterion, Kolmogorov-Smirnov (KS) test statistic, p-value and 
    parameters of the fit distribution.

    Parameters
    ----------
    data
        Data to test fit against.

    Returns
    -------
    tuple[float, float, float, float, float]
        AIC value, KS Statistic and p-value and parameters of the distribution fit.
    """
    data = pd.Series(np.asarray(data, dtype=np.int64))
    zip_model = cm.ZeroInflatedPoisson(endog=data, 
                                       exog=np.ones_like(data)[:, np.newaxis], 
                                       exog_infl=np.ones_like(data)[:, np.newaxis], 
                                       inflation='logit')
    zip_results = zip_model.fit(disp=0)
    lambda_zip = np.exp(zip_results.params["const"])
    pi_zip = 1.0 / (1.0 + np.exp(-zip_results.params["inflate_const"]))

    def cdf_zip(x: Sequence[float]) -> np.ndarray:
        x = np.atleast_1d(x)
        cdf = np.zeros_like(x, dtype=float)
        cdf[x==0] = pi_zip + (1 - pi_zip) * st.poisson.cdf(0, mu=lambda_zip)
        cdf[x>=1] = pi_zip + (1 - pi_zip) * st.poisson.cdf(x[x>=1], mu=lambda_zip)
        return cdf

    ks_stat, p_value = st.kstest(data, cdf_zip)
    return zip_results.aic, ks_stat, p_value, lambda_zip, pi_zip

def test_zero_inflated_negative_binomial_fit(data: Sequence[float]) ->\
                tuple[float, float, float, float, float, float]:
    """
    Determines fit of a zero-inflated negative binomial distribution and returns 
    the Akaike Information Criterion, Kolmogorov-Smirnov (KS) test statistic, 
    p-value and parameters of the fit distribution.

    Parameters
    ----------
    data
        Data to test fit against.

    Returns
    -------
    tuple[float, float, float, float, float, float]
        AIC value, KS Statistic and p-value and parameters of the distribution fit.
    """
    data = pd.Series(np.asarray(data, dtype=np.int64))
    zinb_model = cm.ZeroInflatedNegativeBinomialP(endog=data, 
                                                  exog=np.ones_like(data)[:, np.newaxis], 
                                                  exog_infl=np.ones_like(data)[:, np.newaxis], 
                                                  inflation='logit')
    zinb_results = zinb_model.fit(disp=0)
    alpha, lambda_zinb, gamma = zinb_results.params["alpha"], \
                            np.exp(zinb_results.params["const"]), \
                            zinb_results.params["inflate_const"]
    r = 1.0 / alpha
    p = r / (r + lambda_zinb)
    pi_zinb = 1.0 / (1.0 + np.exp(-gamma))
    
    def cdf_zinb(x: Sequence[float]) -> np.ndarray:
        x = np.atleast_1d(x)
        cdf = np.zeros_like(x, dtype=float)
        cdf[x==0] = pi_zinb + (1 - pi_zinb) * st.nbinom.cdf(0, r, p)
        cdf[x>=1] = pi_zinb + (1 - pi_zinb) * st.nbinom.cdf(x[x>=1], r, p)
        return cdf

    ks_stat, p_value = st.kstest(data, cdf_zinb)
    return zinb_results.aic, ks_stat, p_value, r, p, pi_zinb
