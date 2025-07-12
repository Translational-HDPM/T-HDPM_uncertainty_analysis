"""
Unit tests for functions in the module postprocessing.py.
"""

import pytest
import numpy as np
import pandas as pd

import context
from src.postprocessing import (
    get_differential_classification,
    get_threshold,
    build_sensitivity_specificity_df,
    calculate_jaccard_index,
    calculate_subject_wise_agreement,
    calculate_subject_wise_disagreement,
    calculate_sensitivity_specificity_and_predictive_values,
)

print(context.foo)  # To handle ruff unused imports error


@pytest.fixture
def sample_data_diff_classification():
    gt_labels = pd.Series([0, 0, 1, 1, 0, 1], index=["A", "B", "C", "D", "E", "F"])
    pred_labels_dict = {
        10: pd.Series([0, 1, 1, 0, 0, 1], index=["A", "B", "C", "D", "E", "F"]),
        20: pd.Series([1, 1, 0, 0, 1, 0], index=["A", "B", "C", "D", "E", "F"]),
        30: pd.Series([0, 0, 0, 0, 0, 0], index=["A", "B", "C", "D", "E", "F"]),
    }
    labels = ["NCI", "AD"]
    return gt_labels, pred_labels_dict, labels


def test_get_differential_classification(sample_data_diff_classification):
    """Test get_differential_classification with a basic scenario."""
    gt_labels, pred_labels_dict, labels = sample_data_diff_classification

    expected_data = {
        "NCI": [0.0, 33.33333333333333, 100.0, 0.0],
        "AD": [0.0, 33.33333333333333, 100.0, 100.0],
    }
    expected_df = pd.DataFrame(expected_data, index=[0, 10, 20, 30])
    expected_df.index.name = None  # Ensure index name matches if not set by function

    result_df = get_differential_classification(gt_labels, pred_labels_dict, labels)

    # Sort indices for comparison as the function sorts them
    result_df = result_df.loc[sorted(result_df.index)]
    expected_df = expected_df.loc[sorted(expected_df.index)]

    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=True, atol=1e-6)


@pytest.fixture
def sample_data_sens_spec():
    gt_labels = pd.Series([0, 1, 0, 1, 0], index=[0, 1, 2, 3, 4])  # 3 NCI (0), 2 AD (1)
    pred_labels = pd.Series(
        [0, 0, 1, 1, 0], index=[0, 1, 2, 3, 4]
    )  # Predicted: NCI, NCI, AD, AD, NCI
    return gt_labels, pred_labels


def test_calculate_sensitivity_specificity_and_predictive_values(sample_data_sens_spec):
    """Test calculate_sensitivity_specificity_and_predictive_values for a specific label."""
    gt_labels, pred_labels = sample_data_sens_spec

    # Test for label_idx = 1 (AD)
    sens_ad, spec_ad, ppv_ad, npv_ad = (
        calculate_sensitivity_specificity_and_predictive_values(
            gt_labels, pred_labels, label_idx=1
        )
    )
    np.testing.assert_allclose(sens_ad, 0.5, atol=1e-6)
    np.testing.assert_allclose(spec_ad, 2 / 3, atol=1e-6)  # 0.666...
    np.testing.assert_allclose(ppv_ad, 0.5, atol=1e-6)
    np.testing.assert_allclose(npv_ad, 2 / 3, atol=1e-6)  # 0.666...

    # Test for label_idx = 0 (NCI)
    sens_nci, spec_nci, ppv_nci, npv_nci = (
        calculate_sensitivity_specificity_and_predictive_values(
            gt_labels, pred_labels, label_idx=0
        )
    )
    np.testing.assert_allclose(sens_nci, 2 / 3, atol=1e-6)  # 0.666...
    np.testing.assert_allclose(spec_nci, 0.5, atol=1e-6)
    np.testing.assert_allclose(ppv_nci, 2 / 3, atol=1e-6)  # 0.666...
    np.testing.assert_allclose(npv_nci, 0.5, atol=1e-6)

    # Test with invalid label_idx
    with pytest.raises(ValueError, match="Label not found"):
        calculate_sensitivity_specificity_and_predictive_values(
            gt_labels, pred_labels, label_idx=99
        )


@pytest.fixture
def sample_data_agreement():
    uncertainties = [10, 20]
    n_samples = 5

    gt_series_dict_10 = pd.Series(
        [np.array([0]), np.array([1]), np.array([0])], index=["P0", "P1", "P2"]
    )
    pred_series_dict_10 = pd.Series(
        [
            np.array([0, 0, 1, 0, 0]),
            np.array([1, 1, 0, 1, 1]),
            np.array([1, 1, 1, 1, 1]),
        ],
        index=["P0", "P1", "P2"],
    )

    gt_series_dict_20 = pd.Series(
        [np.array([0]), np.array([1]), np.array([0])], index=["P0", "P1", "P2"]
    )
    pred_series_dict_20 = pd.Series(
        [
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ],
        index=["P0", "P1", "P2"],
    )

    gt_series_dict = {10: gt_series_dict_10, 20: gt_series_dict_20}
    pred_series_dict = {10: pred_series_dict_10, 20: pred_series_dict_20}

    return gt_series_dict, pred_series_dict, uncertainties, n_samples


def test_calculate_subject_wise_agreement(sample_data_agreement):
    """Test calculate_subject_wise_agreement with various agreement levels."""
    gt_series_dict, pred_series_dict, uncertainties, n_samples = sample_data_agreement

    result_df = calculate_subject_wise_agreement(
        gt_series_dict=gt_series_dict,
        pred_series_dict=pred_series_dict,
        uncertainties=uncertainties,
        n_samples=n_samples,
    )

    expected_data = {
        "10% uncertainty": [80.0, 80.0, 0.0],
        "20% uncertainty": [100.0, 0.0, 100.0],
    }
    expected_df = pd.DataFrame(expected_data, index=["P0", "P1", "P2"])
    for col in expected_df.columns:
        expected_df[col] = expected_df[col].astype(object)
    expected_df.index.name = "Patient ID"
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=True, atol=1e-6)


@pytest.fixture
def sample_data_disagreement():
    uncertainties = [10]
    categories = ["NCI", "AD"]  # 0: NCI, 1: AD
    n_samples = 5

    gt_series_dict_10 = pd.Series([np.array([0]), np.array([1])], index=["P0", "P1"])
    pred_series_dict_10 = pd.Series(
        [np.array([0, 0, 1, 0, 0]), np.array([0, 0, 0, 0, 0])], index=["P0", "P1"]
    )

    gt_series_dict = {10: gt_series_dict_10}
    pred_series_dict = {10: pred_series_dict_10}

    return gt_series_dict, pred_series_dict, uncertainties, categories, n_samples


def test_calculate_subject_wise_disagreement(sample_data_disagreement):
    """Test calculate_subject_wise_disagreement with various misclassification scenarios."""
    gt_series_dict, pred_series_dict, uncertainties, categories, n_samples = (
        sample_data_disagreement
    )

    result_df = calculate_subject_wise_disagreement(
        gt_series_dict=gt_series_dict,
        pred_series_dict=pred_series_dict,
        uncertainties=uncertainties,
        categories=categories,
        n_samples=n_samples,
    )

    expected_data = {
        "10% uncertainty: % misclassified as NCI": [
            np.nan,
            100.0,
        ],  # P0: True NCI, P1: Misclassified as NCI 5 times
        "10% uncertainty: % misclassified as AD": [
            20.0,
            np.nan,
        ],  # P0: Misclassified as AD 1 time, P1: True AD
    }
    expected_df = pd.DataFrame(expected_data, index=["P0", "P1"])
    expected_df.index.name = None  # Ensure index name matches if not set by function
    for col in expected_df.columns:
        expected_df[col] = expected_df[col].astype(object)
    # Need to handle NaN comparison. pd.testing.assert_frame_equal handles NaN by default.
    pd.testing.assert_frame_equal(
        result_df, expected_df, check_dtype=True, atol=1e-2
    )  # atol for np.round(..., 2)


@pytest.fixture
def sample_data_build_sens_spec_df():
    gt_probs_ser = pd.Series([0.0, 0.99, 0.4, 0.7, 0.0, 0.8], index=[0, 1, 2, 3, 4, 5])
    pathos_data = {
        "Disease": ["NCI", "AD", "NCI", "AD", "NCI", "AD"],
        "OtherFeature": [1, 2, 3, 4, 5, 6],
    }
    pathos_df = pd.DataFrame(pathos_data, index=[0, 1, 2, 3, 4, 5])
    return pathos_df, gt_probs_ser


def test_build_sensitivity_specificity_df(sample_data_build_sens_spec_df):
    """Test build_sensitivity_specificity_df for 'AD' label."""
    pathos_df, gt_probs_ser = sample_data_build_sens_spec_df

    # Test for label 'AD'
    result_df_ad = build_sensitivity_specificity_df(pathos_df, gt_probs_ser, "AD")
    assert "threshold" in result_df_ad.columns
    assert "sensitivity" in result_df_ad.columns
    assert "specificity" in result_df_ad.columns
    assert "ppv" in result_df_ad.columns
    assert "npv" in result_df_ad.columns
    assert not result_df_ad.empty
    assert result_df_ad.shape[0] == 99  # thresholds from 1 to 99

    threshold_50_row = result_df_ad[result_df_ad["threshold"] == 50]
    assert not threshold_50_row.empty
    np.testing.assert_allclose(threshold_50_row["sensitivity"].iloc[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(threshold_50_row["specificity"].iloc[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(threshold_50_row["ppv"].iloc[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(threshold_50_row["npv"].iloc[0], 1.0, atol=1e-6)

    # Test for invalid label
    with pytest.raises(ValueError, match="Invalid label."):
        build_sensitivity_specificity_df(pathos_df, gt_probs_ser, "Invalid")


@pytest.fixture
def sample_data_get_threshold():
    # Create a sample sens_spec_df
    data = {
        "threshold": np.arange(1, 100),
        "sensitivity": np.linspace(0.99, 0.01, 99),  # Decreasing sensitivity
        "specificity": np.linspace(0.01, 0.99, 99),  # Increasing specificity
    }
    sens_spec_df = pd.DataFrame(data)
    return sens_spec_df


def test_get_threshold_sensitivity_decreasing(sample_data_get_threshold):
    """Test get_threshold for decreasing sensitivity."""
    sens_spec_df = sample_data_get_threshold

    threshold = get_threshold(50.0, sens_spec_df, "sensitivity")
    assert threshold == 50.0


def test_get_threshold_specificity_increasing(sample_data_get_threshold):
    """Test get_threshold for increasing specificity."""
    sens_spec_df = sample_data_get_threshold

    threshold = get_threshold(50.0, sens_spec_df, "specificity")
    assert threshold == 50.0


def test_get_threshold_value_out_of_bounds(sample_data_get_threshold):
    """Test get_threshold raises ValueError for out-of-bounds value."""
    sens_spec_df = sample_data_get_threshold
    with pytest.raises(ValueError, match="sensitivity out of bounds"):
        get_threshold(100.0, sens_spec_df, "sensitivity")
    with pytest.raises(ValueError, match="specificity out of bounds"):
        get_threshold(0.0, sens_spec_df, "specificity")


def test_get_threshold_invalid_metric(sample_data_get_threshold):
    """Test get_threshold raises ValueError for invalid metric."""
    sens_spec_df = sample_data_get_threshold
    with pytest.raises(
        ValueError, match="Metric must be either sensitivity or specificity."
    ):
        get_threshold(50.0, sens_spec_df, "accuracy")


@pytest.fixture
def sample_data_jaccard():
    labels = ["ClassA", "ClassB"]  # 0: ClassA, 1: ClassB
    gt_labels = pd.Series([0, 0, 1, 1, 0], index=["s1", "s2", "s3", "s4", "s5"])
    pred_labels_dict = {
        10: pd.Series(
            [0, 1, 1, 0, 0], index=["s1", "s2", "s3", "s4", "s5"]
        ),  # Mixed predictions
        20: pd.Series(
            [0, 0, 1, 1, 0], index=["s1", "s2", "s3", "s4", "s5"]
        ),  # Perfect match
        30: pd.Series(
            [1, 1, 0, 0, 1], index=["s1", "s2", "s3", "s4", "s5"]
        ),  # Inverse predictions
    }
    return labels, gt_labels, pred_labels_dict


def test_calculate_jaccard_index(sample_data_jaccard):
    """Test calculate_jaccard_index with various prediction scenarios."""
    labels, gt_labels, pred_labels_dict = sample_data_jaccard

    result_df = calculate_jaccard_index(
        labels=labels, gt_labels=gt_labels, pred_labels_dict=pred_labels_dict
    )

    expected_data = {"ClassA": [0.5, 1.0, 0.0, 1.0], "ClassB": [1 / 3, 1.0, 0.0, 1.0]}
    expected_df = pd.DataFrame(expected_data, index=[10, 20, 30, 0])
    expected_df.index.name = None  # Ensure index name matches if not set by function
    for col in expected_df.columns:
        expected_df[col] = expected_df[col].astype(object)

    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=True, atol=1e-6)
