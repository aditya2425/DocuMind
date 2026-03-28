"""Tests for the dataset management module."""

import json
import tempfile
from pathlib import Path

from src.evaluation.dataset import (
    SAMPLE_DATASET,
    load_dataset,
    merge_datasets,
    save_dataset,
)


def test_sample_dataset_is_valid():
    """Built-in sample dataset should have required fields."""
    assert len(SAMPLE_DATASET) >= 5
    for item in SAMPLE_DATASET:
        assert "question" in item
        assert "ground_truth" in item


def test_load_missing_file_returns_sample():
    """Loading a non-existent file should return the sample dataset."""
    dataset = load_dataset("/nonexistent/path/dataset.json")
    assert len(dataset) == len(SAMPLE_DATASET)


def test_save_and_load_roundtrip():
    """Saving and loading should produce identical data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test_dataset.json")
        save_dataset(SAMPLE_DATASET, path)
        loaded = load_dataset(path)
        assert len(loaded) == len(SAMPLE_DATASET)
        assert loaded[0]["question"] == SAMPLE_DATASET[0]["question"]


def test_merge_deduplicates():
    """Merging datasets should remove duplicate questions."""
    ds1 = [{"question": "What is AI?", "ground_truth": "Answer 1"}]
    ds2 = [{"question": "What is AI?", "ground_truth": "Answer 2"}]
    ds3 = [{"question": "What is ML?", "ground_truth": "Answer 3"}]
    merged = merge_datasets(ds1, ds2, ds3)
    assert len(merged) == 2
