from TeleOCR.version import __version__


def test_runtime_version_matches_release():
    assert __version__ == "1.0.0"
