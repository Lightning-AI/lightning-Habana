"""Cofigure local testing."""

import os

_TEST_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)
_TEMP_PATH = os.path.join(_PROJECT_ROOT, "test_temp")
_PATH_DATASETS = os.path.join(_PROJECT_ROOT, "Datasets")

from lightning_habana import _HPU_AVAILABLE  # noqa: E402

assert _HPU_AVAILABLE
