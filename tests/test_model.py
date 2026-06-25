"""Unit tests for SentimentScope — run with: pytest tests/"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from aspect_extractor import extract_aspects, _keyword_fallback
from data_loader import make_demo_dataset, train_val_split

def test_keyword_fallback_finds_battery():
    result = _keyword_fallback("The battery life is excellent")
    assert "battery" in result

def test_keyword_fallback_finds_multiple():
    result = _keyword_fallback("The screen and camera are great")
    assert "screen" in result
    assert "camera" in result

def test_keyword_fallback_no_match_returns_product():
    result = _keyword_fallback("This thing is amazing")
    assert result == ["product"]

def test_extract_aspects_returns_list():
    result = extract_aspects("The battery is great but screen is bad")
    assert isinstance(result, list)
    assert len(result) > 0

def test_demo_dataset_structure():
    ds = make_demo_dataset(50)
    assert len(ds) == 50
    for sample in ds:
        assert "text"   in sample
        assert "aspect" in sample
        assert "label"  in sample
        assert sample["label"] in ("positive", "neutral", "negative")

def test_train_val_split_ratio():
    ds = make_demo_dataset(100)
    train, val = train_val_split(ds, val_ratio=0.2)
    assert len(train) == 80
    assert len(val)   == 20

def test_train_val_split_no_overlap():
    ds = make_demo_dataset(100)
    train, val = train_val_split(ds)
    assert len(train) + len(val) == 100

def test_model_loads():
    try:
        from model import SentimentScopeModel
        m = SentimentScopeModel()
        assert m.model is not None
    except ImportError:
        pytest.skip("torch / transformers not installed")

def test_model_predict_shape():
    try:
        from model import SentimentScopeModel
        m = SentimentScopeModel()
        results = m.predict("Battery is great but screen is bad", ["battery", "screen"])
        assert len(results) == 2
        for r in results:
            assert r["sentiment"] in ("positive", "neutral", "negative")
            assert 0.0 <= r["confidence"] <= 1.0
    except ImportError:
        pytest.skip("torch / transformers not installed")
