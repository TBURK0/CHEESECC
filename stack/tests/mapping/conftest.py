import pytest

from cheesestack.mapping import load_sample_map


@pytest.fixture
def sample_map():
    return load_sample_map()
