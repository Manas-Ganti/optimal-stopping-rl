import numpy as np
import pytest

from optstop.config import load_config
from optstop.contracts import Contract


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def put():
    return Contract("put", 40.0, 1.0, 50, 0.06)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
