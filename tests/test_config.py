import pytest

import TeleOCR.config as config


def test_update_is_atomic(monkeypatch):
    monkeypatch.setattr(config, "MAX_MODEL_LEN", 16384)
    with pytest.raises(ValueError, match="GPU_MEMORY_UTILIZATION"):
        config.update(["MAX_MODEL_LEN=2048", "GPU_MEMORY_UTILIZATION=2"])
    assert config.MAX_MODEL_LEN == 16384


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("BACKEND=http-client", "Unsupported BACKEND"),
        ("LAYOUT_MODE=Other", "Unsupported LAYOUT_MODE"),
        ("PDF_TOOLS=Other", "Unsupported PDF_TOOLS"),
        ("MAX_PIXELS=0", "MAX_PIXELS"),
        ("MAX_NEW_TOKENS=0", "MAX_NEW_TOKENS"),
    ],
)
def test_invalid_configuration_is_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        config.update([override])


def test_unknown_override_is_rejected():
    with pytest.raises(KeyError, match="Unknown config option"):
        config.update(["NOT_A_SETTING=1"])
