from lightning_habana.utils.resources import get_hpu_synapse_version


def test_get_version():
    ver = get_hpu_synapse_version()
    assert ver.startswith("1.")
