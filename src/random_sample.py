"""
Functions to draw random samples from specific probability distributions.
"""

import numpy as np

def sample_zero_inflated_poisson(
        lambda_zip: float, pi_zip: float, shape: tuple[int, int]) -> np.ndarray:
    """
    Generates random samples of given shape from a zero-inflated Poisson distribution.

    Parameters
    ----------
    lambda_zip
        Expected value of the distribution, or the $\lambda$ parameter
    pi_zip
        Probability of extra zeros.
    shape
        Dimensions of the output sample array

    Returns
    -------
    np.ndarray
        Random sample array of given shape.
    """
    probs = np.random.rand(*shape)
    res = np.random.poisson(lam=lambda_zip, size=shape)
    res[probs < pi_zip] = 0.0
    return res

def sample_zero_inflated_negative_binomial(
        r: float, p: float, pi_zinb: float, shape: tuple[int, int]) -> np.ndarray:
    """
    Generates random samples of given shape from a zero-inflated negative binomial distribution.

    Parameters
    ----------
    r
        Number of successes until the experiment is stopped.
    p
        Success probability in each experiment.
    pi_zinb
        Probability of extra zeros.
    shape
        Dimensions of the output sample array

    Returns
    -------
    np.ndarray
        Random sample array of given shape.
    """
    probs = np.random.rand(*shape)
    res = np.random.negative_binomial(r, p, size=shape)
    res[probs < pi_zinb] = 0.0
    return res

def sample_gaussian_mean_rsd(mu: float | np.ndarray, rel_u: float, n_points: int = 1000) -> np.ndarray:
    """
    Given a mean $X$ and uncertainty % (relative standard deviation) $k$, sample from 
    a Gaussian distribution corresponding to these.

    Parameters
    ----------
    mu
        Mean of resulting distribution
    rel_u
        Relative standard deviation of the distribution
    n_points
        Number of points to sample

    Returns
    -------
    np.ndarray
        Random sample array of given shape.

    Raises
    ------
    ValueError:
        Relative uncertainty must be between 0 and 1.

    """
    if not 0 <= rel_u <= 1:
        raise ValueError("Relative uncertainty must be between 0 and 1.")
    return np.random.normal(mu, rel_u * mu, size=n_points)

def sample_poisson_mean_rsd(mu: int, rel_u: float, n_points: int = 1000) -> np.ndarray:
    r"""
    Given a mean $X$ and uncertainty % (relative standard deviation) $k$, sample from 
    a Poisson distribution corresponding to these.
    For a Poisson distribution the relative uncertainty (i.e. the standard deviation
    divided by the mean) is

    $$
    \frac{\sigma}{\mu} = \frac{\sqrt{\lambda}}{\lambda} = \frac{1}{\sqrt{\lambda}}.
    $$
    
    So if we want a relative uncertainty of $k$, we need
    
    $$
    \frac{1}{\sqrt{\lambda}} = k \quad \Longrightarrow \quad \lambda = \frac{1}{k^2}.
    $$
    
    To achieve this we sample from a Poisson distribution with $\lambda = \frac{1}{k^2}$. 
    Then we scale each value by $\frac{X}{1/k^2} = Xk^2$ so that the final samples 
    have a mean $X$ and standard deviation $kX$.

    Parameters
    ----------
    mu
        Mean of resulting distribution
    rel_u
        Relative standard deviation of the distribution
    n_points
        Number of points to sample

    Returns
    -------
    np.ndarray
        Random sample array of given shape.

    Raises
    ------
    ValueError:
        Relative uncertainty must be between 0 and 1.
    """
    if not 0 <= rel_u <= 1:
        raise ValueError("Relative uncertainty must be between 0 and 1.")
    return mu * (rel_u ** 2) * np.random.poisson(lam=1 / rel_u**2.0, size=n_points)

def sample_zero_inflated_poisson_mean_rsd(
        mu: int, rel_u: float, n_points: int = 1000) -> np.ndarray:
    r"""
    Given a mean $X$ and uncertainty % (relative standard deviation) $k$, sample from 
    a zero-inflated Poisson distribution corresponding to these.
    For a ZIP model, the probability mass function is

    $$
    P(Y=0) = \pi + (1-\pi)e^{-\lambda},
    \quad P(Y=y) = (1-\pi)e^{-\lambda}\frac{\lambda^y}{y!}\quad \text{for } y\ge1.
    $$
    
    where $\pi$ is the probability threshold below which a random sample is a zero.
    
    Its mean $\mu$ and standard deviation $\sigma$ are
    
    $$
    \mu = (1-\pi)\lambda
    $$
    
    $$
    \sigma = \sqrt{\lambda(1-\pi)(1+\pi\lambda)}
    $$
    
    If we want $\mu = X$ and $\sigma = kX$, we have
    
    $$
    X = (1-\pi)\lambda
    $$
    
    $$
    kX = \sqrt{\lambda(1-\pi)(1+\pi\lambda)}
    $$

    Solving for $\pi$ and $\lambda$ gives
    
    $$
    \pi = \frac{1}{1+\frac{1}{k^2-1/\mu}}
    $$
    
    $$
    \lambda = \frac{X}{1-\pi}
    $$
    
    Here as well, as $0 \le \pi \le 1$, we need $Xk^2 > 1$. 

    Parameters
    ----------
    mu
        Mean of resulting distribution
    rel_u
        Relative standard deviation of the distribution
    n_points
        Number of points to sample

    Returns
    -------
    np.ndarray
        Random sample array of given shape.

    Raises
    ------
    ValueError:
        Relative uncertainty must be between 0 and 1.
    ValueError:
        For the given relative uncertainty rel_u, we need rel_u^2 > 1 / mean.
    """
    if not 0 <= rel_u <= 1:
        raise ValueError("Relative uncertainty must be between 0 and 1.")
    if rel_u ** 2.0 <= 1/mu:
        raise ValueError("For the given relative uncertainty rel_u, we need rel_u^2 > 1 / mean")
    pi_zip = 1.0 / (1.0 + 1.0 / (rel_u ** 2.0 - 1.0 / mu))
    probs = np.random.rand(n_points)
    res = np.random.poisson(lam=mu / (1-pi_zip), size=n_points)
    res[probs < pi_zip] = 0.0
    return res

def sample_negative_binomial_mean_rsd(mu: int, rel_u: float, n_points: int = 1000) -> np.ndarray:
    r"""
    Given a mean ($X$) and uncertainty % (relative standard deviation) $k$, sample from 
    a negative binomial distribution corresponding to these.
    For a negative binomial distribution $NB(r, p)$ parametrized by $r$ (number of successes 
    until the experiment is stopped) and $p$ (success probability in each experiment), 
    mean $\mu$ and standard deviation $\sigma$ are given by

    $$
    \mu = \frac{r(1-p)}{p}
    $$

    $$
    \sigma = \frac{\sqrt{r(1-p)}}{p}
    $$

    So, we need

    $$
    X = \frac{r(1-p)}{p}
    $$

    $$
    kX = \frac{\sqrt{r(1-p)}}{p}
    $$

    When we solve these two equations for $r$ and $p$, we get

    $$
    p = \frac{1}{Xk^2}
    $$

    $$
    r = \frac{1}{(1-p)}{k^2}
    $$

    Because $0 \le p \le 1$, we must have

    $$
    k^2 > \frac{1}{X}.
    $$

    In other words, the desired relative uncertainty $k$ must be at least as large as
    the Poisson limit $1/\sqrt{X}$.
    (For instance, if $X=50$, then $1/\sqrt{50}\approx 14\%$; we cannot model a 10%
    relative uncertainty with a negative binomial when the “baseline” Poisson uncertainty
    is 14%.)

    Parameters
    ----------
    mu
        Mean of resulting distribution
    rel_u
        Relative standard deviation of the distribution
    n_points
        Number of points to sample

    Returns
    -------
    np.ndarray
        Random sample array of given shape.


    Raises
    ------
    ValueError:
        Relative uncertainty must be between 0 and 1.
    ValueError:
        For the given relative uncertainty rel_u, we need rel_u^2 > 1 / mean.
    """
    if not 0 <= rel_u <= 1:
        raise ValueError("Relative uncertainty must be between 0 and 1.")
    if rel_u ** 2.0 <= 1/mu:
        raise ValueError("For the given relative uncertainty rel_u, we need rel_u^2 > 1 / mean")
    p = 1 / (mu * rel_u ** 2)
    r = 1 / ((1 - p) * rel_u ** 2)
    return np.random.negative_binomial(r, p, size=n_points)

def sample_zero_inflated_negative_binomial_mean_rsd(
        mu: int, rel_u: float, pi_zinb: float, n_points: int = 1000) -> np.ndarray:
    r"""
    Given a mean ($X$) and uncertainty % (relative standard deviation) $k$, sample 
    from a zero-inflated negative binomial distribution corresponding to these.
    
    We use a similar parametrization as the negative binomial distribution above with 
    the added parameter of $\pi$. 
    Since this time $\pi$ can be set independently of $X$ and $k$ we can simply set
    $$
    p = \frac{1}{Xk^2} 
    $$
    
    $$
    r = \frac{1}{(1-p)}{k^2}
    $$
    
    Because $0 \le p \le 1$, we must have
    
    $$
    k^2 > \frac{1}{X}.
    $$

    Parameters
    ----------
    mu
        Mean of resulting distribution
    rel_u
        Relative standard deviation of the distribution
    pi_zinb
        Probability of extra zeros.
    n_points
        Number of points to sample

    Returns
    -------
    np.ndarray
        Random sample array of given shape.


    Raises
    ------
    ValueError:
        Relative uncertainty must be between 0 and 1.
    ValueError:
        Probability must be between 0 and 1.
    ValueError:
        For the given relative uncertainty rel_u, we need rel_u^2 > 1 / mean.

    """
    if not 0 <= rel_u <= 1:
        raise ValueError("Relative uncertainty must be between 0 and 1.")
    if not 0 <= pi_zinb <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if rel_u ** 2.0 <= 1/mu:
        raise ValueError("For the given relative uncertainty rel_u, we need rel_u^2 > 1 / mean")
    p = 1 / (mu * rel_u ** 2)
    r = 1 / ((1 - p) * rel_u ** 2)
    probs = np.random.rand(n_points)
    res = np.random.negative_binomial(r, p, size=n_points)
    res[probs < pi_zinb] = 0.0
    return res

def sample_gamma_mean_rsd(mu: float, rel_u: float, n_points: int) -> np.ndarray:
    """
    Given a mean and uncertainty % (relative standard deviation), sample from a Gamma 
    distribution corresponding to these.

    Parameters
    ----------
    mu
        Mean of resulting distribution
    rel_u
        Relative standard deviation of the distribution
    n_points
        Number of points to sample

    Returns
    -------
    np.ndarray
        Random sample array of given shape.


    Raises
    ------
    ValueError:
        Relative uncertainty must be between 0 and 1.
    """
    if not 0 <= rel_u <= 1:
        raise ValueError("Relative uncertainty must be between 0 and 1.")
    if mu == 0:
        mu += 1.0
    return np.random.gamma(1 / (rel_u ** 2), mu * rel_u ** 2, size=n_points)

def sample_lognormal_mean_rsd(mu: float, rel_u: float, n_points: int) -> np.ndarray:
    """
    Given a mean and uncertainty % (relative standard deviation), sample from a 
    log normal distribution corresponding to these.

    Parameters
    ----------
    mu
        Mean of resulting distribution
    rel_u
        Relative standard deviation of the distribution
    n_points
        Number of points to sample

    Returns
    -------
    np.ndarray
        Random sample array of given shape.


    Raises
    ------
    ValueError:
        Relative uncertainty must be between 0 and 1.
    """
    if not 0 <= rel_u <= 1:
        raise ValueError("Relative uncertainty must be between 0 and 1.")
    if mu == 0:
        mu += 1.0
    return np.random.lognormal(mean=np.log(mu) - 0.5 * np.log(1 + rel_u**2),
                               sigma=np.sqrt(np.log(1 + rel_u**2)), size=n_points)
