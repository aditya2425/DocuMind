"""Tests for the experiment runner module."""

import json
import tempfile
from pathlib import Path

from src.evaluation.experiment import (
    _average_scores,
    list_experiments,
    save_results,
    load_results,
)


def test_average_scores():
    """Should correctly average metric scores."""
    metrics = [
        {"scores": {"faithfulness": 0.8, "relevance": 0.6}},
        {"scores": {"faithfulness": 0.6, "relevance": 0.4}},
    ]
    avg = _average_scores(metrics)
    assert abs(avg["faithfulness"] - 0.7) < 0.001
    assert abs(avg["relevance"] - 0.5) < 0.001


def test_average_scores_empty():
    """Empty metrics should return empty dict."""
    assert _average_scores([]) == {}


def test_save_and_load_results():
    """Results should survive a save/load roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        experiment = {
            "experiment_id": "test_001",
            "summary": {"config_a": {"faithfulness": 0.9}},
        }

        # Monkey-patch RESULTS_DIR for test
        import src.evaluation.experiment as exp_mod
        original_dir = exp_mod.RESULTS_DIR
        exp_mod.RESULTS_DIR = tmpdir

        try:
            path = save_results(experiment, "test_001")
            loaded = load_results(path)
            assert loaded["experiment_id"] == "test_001"
            assert loaded["summary"]["config_a"]["faithfulness"] == 0.9
        finally:
            exp_mod.RESULTS_DIR = original_dir


def test_list_experiments_empty():
    """Empty directory should return empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assert list_experiments(tmpdir) == []
