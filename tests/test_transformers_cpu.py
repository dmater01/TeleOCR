import sys
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="requires the Transformers backend extra")

import TeleOCR.config as config  # noqa: E402
from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE  # noqa: E402
from TeleOCR.vlm_utils.vlm_client.transformers_client import (  # noqa: E402
    TransformersVlmClient,
)


class Batch(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[1, 2]]))
        self.input_ids = self["input_ids"]

    def to(self, **kwargs):
        return self


class FakeProcessor:
    tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=0)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def __call__(self, **kwargs):
        return Batch()

    def batch_decode(self, *args, **kwargs):
        return ["ok"]


class FakeModel:
    config = SimpleNamespace(
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        assert torch.is_inference_mode_enabled()
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 9]])

    def eval(self):
        return self


def test_cpu_model_load_preserves_checkpoint_dtype(monkeypatch):
    captured = {}

    class AutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured.update(kwargs)
            return FakeModel()

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = FakeProcessor
    transformers.AutoModelForImageTextToText = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model_path = "cpu-test-model"
    key_prefix = ("transformers", model_path)
    for key in list(TeleOCRMODEL_SERVICE._models):
        if key[:2] == key_prefix:
            del TeleOCRMODEL_SERVICE._models[key]

    TeleOCRMODEL_SERVICE.get_model("transformers", model_path)

    assert captured["dtype"] == "auto"
    assert captured["low_cpu_mem_usage"] is True
    assert "torch_dtype" not in captured


def test_generation_is_bounded_and_uses_inference_mode(monkeypatch):
    model = FakeModel()
    client = TransformersVlmClient(model, FakeProcessor(), use_tqdm=False)
    monkeypatch.setattr(config, "MAX_NEW_TOKENS", 7)

    assert client._predict_one_batch([Image.new("RGB", (8, 8))], ["prompt"], None) == ["ok"]
    assert model.generate_kwargs["max_new_tokens"] == 7
    assert "max_length" not in model.generate_kwargs
