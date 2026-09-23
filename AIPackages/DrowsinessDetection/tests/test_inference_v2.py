#!/usr/bin/env python3
"""Inference contract smoke tests for schema v2 / 3-class checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from feature_contract import DROWSINESS_FEATURE_NAMES, FEATURE_SCHEMA_VERSION  # noqa: E402
from inference import (  # noqa: E402
    SchemaContractError,
    load_checkpoint,
    normalize_windows,
    predict_label,
)
from label_contract import CLASS_TO_IDX, NUM_CLASSES  # noqa: E402

CHECKPOINT = PACKAGE_ROOT / "saved_weights" / "best_accuracy_v2.pt"
LEGACY_SMOKE = PACKAGE_ROOT / "saved_weights" / "schema_v2_smoke.pt"


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="best_accuracy_v2.pt missing")
def test_load_v2_checkpoint_three_classes() -> None:
    model, ckpt, window = load_checkpoint(CHECKPOINT, torch.device("cpu"))
    assert ckpt["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert ckpt["feature_names"] == list(DROWSINESS_FEATURE_NAMES)
    assert ckpt["class_to_idx"] == CLASS_TO_IDX
    assert window == int(ckpt["window_frames"])
    assert next(model.parameters()).device.type == "cpu"
    assert model.training is False
    assert model.num_classes == NUM_CLASSES


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="best_accuracy_v2.pt missing")
def test_normalize_and_predict_shape() -> None:
    model, ckpt, window = load_checkpoint(CHECKPOINT, torch.device("cpu"))
    raw = np.zeros((window, 14), dtype=np.float32)
    raw[:, 0] = 1.0
    raw[:, 4] = 1.0
    raw[:, 5] = 1.0
    raw[:, 6] = 0.25
    raw[:, 7] = 0.25
    raw[:, 10] = 0.5
    raw[:, 11] = 0.5
    raw[:, 12] = 0.5
    raw[:, 13] = 0.5
    norm = normalize_windows(raw, ckpt["_standardizer"])
    assert norm.shape == (window, 14)
    assert torch.isfinite(norm).all()
    out = predict_label(model, norm.unsqueeze(0), torch.device("cpu"))
    assert out["label"] in CLASS_TO_IDX
    assert set(out["probabilities"]) == set(CLASS_TO_IDX)


@pytest.mark.skipif(not LEGACY_SMOKE.is_file(), reason="legacy smoke missing")
def test_reject_five_class_checkpoint() -> None:
    with pytest.raises((SchemaContractError, Exception)):
        load_checkpoint(LEGACY_SMOKE, torch.device("cpu"))
