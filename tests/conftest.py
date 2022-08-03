from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture()
def data_dir() -> Path:
    return Path(TemporaryDirectory(prefix="test-data-dir").name)


@pytest.fixture()
def models_dir() -> Path:
    return Path(TemporaryDirectory(prefix="test-models-dir").name)


# @pytest.fixture(autouse=True)
# def setup_tests():
#     # a fixture that will be automatically called.
#     pass
