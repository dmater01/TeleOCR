import pytest
from types import SimpleNamespace

from TeleOCR.errors import LayoutParseError, PostProcessError
from TeleOCR.vlm_utils import TeleOCR_client as client_module
from TeleOCR.vlm_utils.TeleOCR_client import TeleOCRClient, TeleOCRClientHelper
from TeleOCR.vlm_utils.vlm_client import new_vlm_client


def helper():
    return TeleOCRClientHelper(
        backend="transformers",
        prompts={"default": ""},
        sampling_params={},
        layout_image_size=(32, 32),
        min_image_edge=1,
        max_image_edge_ratio=50,
        simple_post_process=False,
        handle_equation_block=True,
        abandon_list=False,
        abandon_paratext=False,
        debug=False,
        keep_four_numbers=True,
    )


def test_missing_backend_module_is_not_advertised():
    with pytest.raises(ValueError, match="Unsupported backend"):
        new_vlm_client("http-client")


def test_malformed_layout_fails_document():
    with pytest.raises(LayoutParseError, match="layout format"):
        helper().parse_layout_output("not a layout record")


def test_postprocess_failure_is_contextual(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("broken")

    monkeypatch.setattr(client_module, "post_process", fail)
    with pytest.raises(PostProcessError) as exc_info:
        helper().post_process([])
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_page_priorities_are_expanded_to_blocks():
    instance = TeleOCRClient.__new__(TeleOCRClient)
    instance.executor = None
    instance.incremental_priority = False
    instance.batch_layout_detect = lambda images, priority: [
        [SimpleNamespace(content=None), SimpleNamespace(content=None)],
        [SimpleNamespace(content=None)],
    ]

    class FakeHelper:
        def batch_prepare_for_extract(self, *args):
            return [(["a", "b"], ["", ""], [None, None], [0, 1]), (["c"], [""], [None], [0])]

        def batch_post_process(self, executor, blocks):
            return blocks

    class FakeClient:
        def batch_predict(self, images, prompts, params, priorities):
            assert priorities == [10, 10, 20]
            return ["a", "b", "c"]

    instance.helper = FakeHelper()
    instance.client = FakeClient()
    blocks = instance.stepping_two_step_extract([object(), object()], priority=[10, 20])
    assert len(blocks) == 2


@pytest.mark.asyncio
async def test_sync_concurrent_api_rejects_running_loop():
    instance = TeleOCRClient.__new__(TeleOCRClient)

    async def result(*args, **kwargs):
        return []

    instance.aio_concurrent_two_step_extract = result
    with pytest.raises(RuntimeError, match="active event loop"):
        instance.concurrent_two_step_extract([])
