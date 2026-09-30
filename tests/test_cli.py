from argparse import Namespace

import infer


def args(input_dir, output_dir, **overrides):
    values = {
        "image_sub_path": str(input_dir),
        "result_save_path": str(output_dir),
        "override": ["BACKEND=transformers"],
        "use_async": False,
        "overwrite": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_duplicate_stems_are_preflight_error(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "invoice.pdf").touch()
    (input_dir / "invoice.png").touch()
    try:
        infer.get_image_paths(input_dir)
    except ValueError as exc:
        assert "colliding stems" in str(exc)
    else:
        raise AssertionError("duplicate stems were accepted")


def test_partial_failure_returns_one(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.pdf").touch()
    monkeypatch.setattr(infer, "parse_args", lambda: args(input_dir, tmp_path / "out"))
    monkeypatch.setattr(infer, "sync_main", lambda *a, **k: [("one.pdf", RuntimeError())])
    assert infer.main() == 1


def test_bad_input_directory_returns_two(tmp_path, monkeypatch):
    monkeypatch.setattr(
        infer,
        "parse_args",
        lambda: args(tmp_path / "missing", tmp_path / "out"),
    )
    assert infer.main() == 2


def test_async_backend_routes_without_flag(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.pdf").touch()
    monkeypatch.setattr(
        infer,
        "parse_args",
        lambda: args(
            input_dir,
            tmp_path / "out",
            override=["BACKEND=vllm-async-engine"],
        ),
    )

    async def async_success(*args, **kwargs):
        return []

    monkeypatch.setattr(infer, "async_main", async_success)
    monkeypatch.setattr(infer, "sync_main", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    assert infer.main() == 0
