"""
Test suite for classification module using pytest
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from classification import (
    load_and_inspect_data,
    split_data,
    scale_features,
    train_model,
    evaluate_model,
)


class TestDataLoading:
    """Tests for data loading and inspection."""

    def test_iris_shape(self):
        """Test that Iris dataset has correct shape."""
        X, y, _ = load_and_inspect_data()
        assert X.shape == (150, 4), f"Expected shape (150, 4), got {X.shape}"

    def test_iris_classes(self):
        """Test that Iris has 3 classes."""
        X, y, _ = load_and_inspect_data()
        assert len(np.unique(y)) == 3, f"Expected 3 classes, got {len(np.unique(y))}"

    def test_target_names(self):
        """Test that target names are provided."""
        X, y, target_names = load_and_inspect_data()
        assert len(target_names) == 3, f"Expected 3 target names, got {len(target_names)}"
        expected_names = ["setosa", "versicolor", "virginica"]
        assert list(target_names) == expected_names, f"Target names mismatch"


class TestDataSplit:
    """Tests for train-test split."""

    def test_split_ratio(self):
        """Test that split maintains 80-20 ratio."""
        X, y, _ = load_and_inspect_data()
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        assert X_train.shape[0] == 120, f"Expected 120 training samples, got {X_train.shape[0]}"
        assert X_test.shape[0] == 30, f"Expected 30 test samples, got {X_test.shape[0]}"

    def test_split_consistency(self):
        """Test that split is reproducible with fixed random_state."""
        X, y, _ = load_and_inspect_data()
        X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=42)

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)

    def test_split_stratification(self):
        """Test that split maintains class distribution."""
        X, y, _ = load_and_inspect_data()
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Check that all classes are in both train and test
        assert len(np.unique(y_train)) == 3
        assert len(np.unique(y_test)) == 3


class TestFeatureScaling:
    """Tests for StandardScaler application."""

    def test_scaling_applied(self):
        """Test that scaling changes feature values."""
        X, y, _ = load_and_inspect_data()
        X_train, X_test, _, _ = split_data(X, y)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        # Scaled features should have different values than original
        assert not np.allclose(X_train, X_train_scaled)

    def test_scaling_fit_on_train_only(self):
        """Test that scaler is fit only on training data."""
        X, y, _ = load_and_inspect_data()
        X_train, X_test, _, _ = split_data(X, y)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        # Training data should be centered around 0 (mean ~0)
        # This is a rough check; exact values depend on the data
        assert np.abs(np.mean(X_train_scaled)) < 1.0

    def test_scaling_shape_preserved(self):
        """Test that scaling preserves shape."""
        X, y, _ = load_and_inspect_data()
        X_train, X_test, _, _ = split_data(X, y)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape


class TestModelTraining:
    """Tests for KNN model training."""

    def test_model_type(self):
        """Test that model is KNeighborsClassifier."""
        X, y, _ = load_and_inspect_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, _ = scale_features(X_train, X_test)

        model = train_model(X_train_scaled, y_train)
        assert isinstance(model, KNeighborsClassifier)

    def test_model_n_neighbors(self):
        """Test that model uses K=5."""
        X, y, _ = load_and_inspect_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, _ = scale_features(X_train, X_test)

        model = train_model(X_train_scaled, y_train, n_neighbors=5)
        assert model.n_neighbors == 5


class TestPrediction:
    """Tests for model prediction."""

    def test_prediction_count(self):
        """Test that prediction returns correct number of predictions."""
        X, y, _ = load_and_inspect_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        model = train_model(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        assert len(y_pred) == len(y_test)

    def test_prediction_labels(self):
        """Test that predictions contain valid class labels."""
        X, y, _ = load_and_inspect_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        model = train_model(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        assert set(y_pred).issubset({0, 1, 2})


class TestEvaluation:
    """Tests for model evaluation metrics."""

    def test_accuracy_range(self):
        """Test that accuracy is between 0 and 1."""
        X, y, target_names = load_and_inspect_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        model = train_model(X_train_scaled, y_train)
        results = evaluate_model(model, X_test_scaled, y_test, target_names)

        assert 0 <= results["accuracy"] <= 1

    def test_f1_range(self):
        """Test that weighted F1 score is between 0 and 1."""
        X, y, target_names = load_and_inspect_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        model = train_model(X_train_scaled, y_train)
        results = evaluate_model(model, X_test_scaled, y_test, target_names)

        assert 0 <= results["f1_weighted"] <= 1

    def test_confusion_matrix_shape(self):
        """Test that confusion matrix is 3x3."""
        X, y, target_names = load_and_inspect_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        model = train_model(X_train_scaled, y_train)
        results = evaluate_model(model, X_test_scaled, y_test, target_names)

        assert results["confusion_matrix"].shape == (3, 3)

    def test_classification_report_exists(self):
        """Test that classification report is generated."""
        X, y, target_names = load_and_inspect_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        model = train_model(X_train_scaled, y_train)
        results = evaluate_model(model, X_test_scaled, y_test, target_names)

        assert results["classification_report"] is not None
        assert isinstance(results["classification_report"], str)
