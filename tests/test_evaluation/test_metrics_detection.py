import pytest

from src.evaluation.metrics_detection import compute_ap


def test_compute_ap_uses_pr_area_not_legacy_eleven_point_floor():
    assert compute_ap([1.0], [0.01]) == pytest.approx(0.01)


def test_compute_ap_perfect_detector():
    assert compute_ap([1.0, 1.0], [0.5, 1.0]) == pytest.approx(1.0)


def test_compute_ap_empty_curve():
    assert compute_ap([], []) == 0.0
