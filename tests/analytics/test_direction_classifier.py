import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from analytics.direction_classifier import EmaTrendThresholds, classify_trend


def test_classify_trend_strong_up():
    thresholds = EmaTrendThresholds(
        neutral_fast=0.0005,
        neutral_slow=0.0002,
        strong_fast=0.002,
        strong_slow=0.001,
        strong_spread=0.002,
    )
    label, notes = classify_trend(0.003, 0.0015, 0.003, thresholds)
    assert label == "ASC_FUERTE"
    assert "suben" in (notes or "")


def test_classify_trend_strong_down():
    thresholds = EmaTrendThresholds()
    label, _ = classify_trend(-0.003, -0.0015, -0.003, thresholds)
    assert label == "DESC_FUERTE"


def test_classify_trend_soft():
    thresholds = EmaTrendThresholds()
    label, _ = classify_trend(0.0008, 0.0003, 0.0008, thresholds)
    assert label == "ASC_SUAVE"


def test_classify_trend_neutral_when_missing():
    thresholds = EmaTrendThresholds()
    label, notes = classify_trend(None, 0.0, 0.0, thresholds)
    assert label == "NEUTRO"
    assert notes is None
