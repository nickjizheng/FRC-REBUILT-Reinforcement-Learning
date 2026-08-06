"""CPU-only collector health checks."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.rl.collector_cycle_v2 import _true_black_camera_mask


def test_true_black_detector_does_not_reject_flat_but_visible_views():
    rgb = np.full((2, 3, 4, 5, 3), 80, np.uint8)
    rgb[0, 1] = 0

    mask = _true_black_camera_mask(rgb)

    assert mask.shape == (2, 3)
    assert mask[0, 1]
    assert not mask[0, 0]
    assert not mask[1].any()


def test_true_black_detector_requires_near_zero_pixels_not_low_variance():
    rgb = np.zeros((1, 1, 2, 2, 3), np.uint8)
    rgb[0, 0, 0, 0] = 3

    assert not _true_black_camera_mask(rgb).any()
    with pytest.raises(ValueError, match="camera batch"):
        _true_black_camera_mask(np.zeros((1, 2, 3), np.uint8))
