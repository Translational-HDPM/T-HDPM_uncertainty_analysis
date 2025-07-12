"""
Unit tests for the functions in the module logreg_classifier
"""

import context
import numpy as np
import pytest

from src.logreg_classifier import (
    antilogit_classifier_score,
    linear_classifier_score,
    linear_classifier_subscores,
    z_score,
)

print(context.foo)  # To silence ruff unused import checks


@pytest.fixture
def coefficients():
    return np.array([1.0, -2.0, 3.0], dtype=np.float32)


@pytest.fixture
def values_1d():
    return np.array([10.0, 5.0, 2.0], dtype=np.float32)


@pytest.fixture
def values_2d():
    return np.array([[10.0, 5.0, 2.0], [1.0, 2.0, 3.0]], dtype=np.float32).T


def test_linear_classifier_subscores_basic(coefficients, values_1d):
    """Test linear_classifier_subscores with a simple case."""
    neg_sub, pos_sub = linear_classifier_subscores(coefficients, values_1d)
    coefficients_local = coefficients[coefficients.argsort()]
    values_1d_local = values_1d[coefficients.argsort()]
    ans = coefficients_local[:, np.newaxis] * values_1d_local
    np.testing.assert_allclose(
        neg_sub, np.sum(ans[coefficients_local < 0.0], axis=0), atol=1e-6
    )
    np.testing.assert_allclose(
        pos_sub, np.sum(ans[coefficients_local >= 0.0], axis=0), atol=1e-6
    )


def test_linear_classifier_subscores_multiple_samples(coefficients, values_2d):
    """Test linear_classifier_subscores with multiple samples."""
    coefficients_local = coefficients[coefficients.argsort()]
    values_2d_local = values_2d[coefficients.argsort(), :]
    ans = coefficients_local[:, np.newaxis] * values_2d_local
    neg_sub, pos_sub = linear_classifier_subscores(coefficients, values_2d)
    np.testing.assert_allclose(
        neg_sub, np.sum(ans[coefficients_local < 0.0, :], axis=0), atol=1e-6
    )
    np.testing.assert_allclose(
        pos_sub, np.sum(ans[coefficients_local >= 0.0, :], axis=0), atol=1e-6
    )


def test_linear_classifier_subscores_all_positive_coeffs(coefficients, values_1d):
    """Test linear_classifier_subscores with all positive coefficients."""
    coefficients_local = np.abs(coefficients)
    coefficients_local = coefficients_local[coefficients_local.argsort()]
    values_1d_local = values_1d[coefficients_local.argsort()]
    ans = coefficients_local[:, np.newaxis] * values_1d_local
    neg_sub, pos_sub = linear_classifier_subscores(coefficients_local, values_1d)
    np.testing.assert_allclose(
        neg_sub, np.sum(ans[coefficients_local < 0.0, :], axis=0), atol=1e-6
    )
    np.testing.assert_allclose(
        pos_sub, np.sum(ans[coefficients_local >= 0.0, :], axis=0), atol=1e-6
    )


def test_linear_classifier_subscores_all_negative_coeffs(coefficients, values_1d):
    """Test linear_classifier_subscores with all negative coefficients."""
    coefficients_local = -np.abs(coefficients)
    coefficients_local = coefficients_local[coefficients_local.argsort()]
    values_1d_local = values_1d[coefficients_local.argsort()]
    ans = coefficients_local[:, np.newaxis] * values_1d_local
    neg_sub, pos_sub = linear_classifier_subscores(coefficients_local, values_1d)
    np.testing.assert_allclose(
        neg_sub, np.sum(ans[coefficients_local < 0.0, :], axis=0), atol=1e-6
    )
    np.testing.assert_allclose(
        pos_sub, np.sum(ans[coefficients_local >= 0.0, :], axis=0), atol=1e-6
    )


def test_linear_classifier_score_basic(coefficients, values_1d):
    """Test linear_classifier_score with a simple case."""
    score = linear_classifier_score(coefficients, values_1d)
    np.testing.assert_allclose(
        score, np.sum(coefficients * values_1d, axis=0), atol=1e-6
    )


def test_linear_classifier_score_multiple_samples(coefficients, values_2d):
    """Test linear_classifier_score with multiple samples."""
    score = linear_classifier_score(coefficients, values_2d)
    np.testing.assert_allclose(
        score, np.sum(coefficients[:, np.newaxis] * values_2d, axis=0), atol=1e-6
    )


def test_antilogit_classifier_score_basic():
    """Test antilogit_classifier_score with a simple linear score."""
    linear_score = 0.0
    prob = antilogit_classifier_score(linear_score)
    np.testing.assert_allclose(prob, 0.5, atol=1e-6)


def test_antilogit_classifier_score_with_gamma():
    """Test antilogit_classifier_score with gamma."""
    linear_score = 1.0
    gamma = -1.0
    prob = antilogit_classifier_score(linear_score, gamma)
    np.testing.assert_allclose(prob, 0.5, atol=1e-6)


def test_antilogit_classifier_score_array_input():
    """Test antilogit_classifier_score with array input."""
    linear_score = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    prob = antilogit_classifier_score(linear_score)
    expected_prob = 1 / (1 + np.exp(-linear_score))
    np.testing.assert_allclose(prob, expected_prob, atol=1e-6)


def test_z_score_scalar():
    """Test z_score with scalar inputs."""
    x = 10.0
    mean = 5.0
    std = 2.0
    z = z_score(x, mean, std)
    np.testing.assert_allclose(z, (10.0 - 5.0) / 2.0, atol=1e-6)  # 2.5


def test_z_score_array():
    """Test z_score with array inputs."""
    x = np.array([10.0, 12.0], dtype=np.float32)
    mean = np.array([5.0, 6.0], dtype=np.float32)
    std = np.array([2.0, 3.0], dtype=np.float32)
    z = z_score(x, mean, std)
    expected_z = np.array([(10.0 - 5.0) / 2.0, (12.0 - 6.0) / 3.0], dtype=np.float32)
    np.testing.assert_allclose(z, expected_z, atol=1e-6)


def test_z_score_zero_std_scalar():
    """Test z_score with zero standard deviation (scalar)."""
    with pytest.raises(ZeroDivisionError):
        z_score(10.0, 5.0, 0.0)


def test_z_score_zero_std_array():
    """Test z_score with zero standard deviation (array)."""
    x = np.array([10.0, 12.0], dtype=np.float32)
    mean = np.array([5.0, 6.0], dtype=np.float32)
    std = np.array([0.0, 0.0], dtype=np.float32)
    with pytest.raises(ZeroDivisionError):
        z_score(x, mean, std)
