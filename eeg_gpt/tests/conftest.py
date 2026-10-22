import os
import shutil
import tempfile
import pytest

from eeg_gpt.tests.test_pipeline import make_corpus


@pytest.fixture(scope="session")
def _corpus():
    root = tempfile.mkdtemp(prefix="eeggpt_pytest_corpus_")
    ck, csv = make_corpus(root, n_patients=3, sessions_per_patient=3, dur_s=600.0)
    yield {"root": root, "ck": ck, "csv": csv}
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def root(tmp_path):
    return str(tmp_path)


@pytest.fixture
def ck(_corpus):
    return _corpus["ck"]


@pytest.fixture
def csv(_corpus):
    return _corpus["csv"]
