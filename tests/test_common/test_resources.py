import pytest
from lightning_habana.utils.resources import device_count, get_gaudi_version


@pytest.mark.skipif(device_count() <= 1, reason="Test requires multiple HPU devices")
def test_get_version():
    ver = get_gaudi_version()
    assert ver.startswith("1.")
