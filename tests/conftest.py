"""Test fixtures — isolate tests from real data and external services."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def isolated_data_root():
    """Redirect .slide_tutor_data to a temp dir so tests never touch real data."""
    with TemporaryDirectory(prefix="slide-tutor-test-") as tmp:
        os.environ["SLIDE_TUTOR_TEST_DATA_ROOT"] = tmp
        # Force config to use the temp dir
        import slide_tutor.config as cfg
        old_root = cfg.DATA_ROOT
        old_projects = cfg.PROJECTS_DIR
        old_modules = cfg.MODULES_DIR
        cfg.DATA_ROOT = Path(tmp)
        cfg.PROJECTS_DIR = cfg.DATA_ROOT / "projects"
        cfg.MODULES_DIR = cfg.DATA_ROOT / "modules"
        cfg.ensure_data_root()
        yield Path(tmp)
        cfg.DATA_ROOT = old_root
        cfg.PROJECTS_DIR = old_projects
        cfg.MODULES_DIR = old_modules
