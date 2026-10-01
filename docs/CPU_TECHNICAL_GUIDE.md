# TeleOCR CPU Technical Guide

This document describes the design, configuration, runtime behavior, tests,
and maintenance model of TeleOCR's Transformers CPU path. For operator-facing
instructions, see the [CPU User Guide](CPU_USER_GUIDE.md).

## 1. Scope and version lineage

The CPU work is layered on the hardened `1.0.0` codebase:

| Reference | Purpose |
| --- | --- |
| `cpu-test-baseline-1.0.0` / `52165ca` | Frozen hardening and reproducibility baseline. |
| `f189195` | Native-dtype CPU loading, low-memory load, inference mode, and bounded generation. |
| `28a08b4` | Optional-backend test compatibility with core-only CI jobs. |

The baseline tag remains fixed. The CPU runtime improvements live on `main`.

## 2. High-level architecture

The CLI pipeline is:

```text
input directory
    |
    v
infer.py: preflight, configuration, deterministic file ordering
    |
    v
read_fn(): content detection and image-to-PDF normalization
    |
    v
engine.do_parse(): page selection and staged output transaction
    |
    v
doc_analyze(): PDF rasterization and model acquisition
    |
    v
TeleOCRClient: layout inference -> block crops -> content inference
    |
    v
middle JSON conversion and post-processing
    |
    v
Markdown + JSON + annotated layout PDF + extracted images
```

CPU execution is synchronous at the CLI level because `BACKEND=transformers`
routes to `sync_main()`. Asynchronous wrappers use worker threads but do not
turn CPU inference into vLLM-style concurrent generation.

## 3. Module responsibilities

| Module | Responsibility |
| --- | --- |
| `infer.py` | Argument parsing, validated overrides, input preflight, backend routing, exit status. |
| `TeleOCR/config.py` | Process-global runtime defaults, conversion, and validation. |
| `TeleOCR/tools/read_file.py` | Content-based file recognition and image-to-PDF conversion. |
| `TeleOCR/engine.py` | Page selection, transactional output staging, Markdown/JSON/layout emission. |
| `TeleOCR/src/vlm_analyze.py` | PDF rasterization, predictor initialization, and page inference orchestration. |
| `TeleOCR/vlm_utils/TeleOCR_model.py` | Singleton model service and backend-specific model construction. |
| `TeleOCR/vlm_utils/TeleOCR_client.py` | Two-stage layout/content extraction and block handling. |
| `TeleOCR/vlm_utils/vlm_client/transformers_client.py` | Prompt formatting, tensor preparation, bounded Transformers generation, decoding. |
| `TeleOCR/src/model_output_to_middle_json.py` | Conversion from model blocks to the stable intermediate document representation. |

## 4. CPU model loading

`TeleOCRMODEL_SERVICE.get_model()` lazily imports PyTorch and Transformers only
when the Transformers backend is selected. This keeps the core package usable
without optional ML frameworks.

The CPU-specific load sequence is:

1. Load the model processor with `trust_remote_code=True`.
2. Detect CUDA availability once.
3. Preserve an explicitly supplied legacy `torch_dtype` by translating it to
   the current `dtype` keyword.
4. Otherwise use `dtype="auto"` on CPU so the checkpoint controls its dtype.
5. Set `low_cpu_mem_usage=True` unless the caller overrides it.
6. Load the model and set evaluation mode.
7. Move the model to CUDA only when CUDA is available.

The TeleOCR checkpoint declares bfloat16. Preserving its native dtype avoids
the former CPU default of float32, which could approximately double weight
memory. On the validated system, bfloat16 CPU matrix multiplication and full
model inference both completed successfully.

The model service caches clients in process memory. The key includes backend,
model path, vLLM context settings, GPU utilization, and explicit keyword
arguments. Repeated documents in one process reuse the loaded model.

## 5. CPU generation safeguards

The Transformers client applies two key safeguards.

### Inference mode

`model.generate()` runs inside `torch.inference_mode()`. This disables gradient
tracking and its associated autograd bookkeeping. It reduces unnecessary
runtime memory and is appropriate because OCR is inference-only.

### Bounded generation

If a request-specific `SamplingParams.max_new_tokens` is provided, it wins.
Otherwise the client uses:

```python
min(CONFIG.MAX_NEW_TOKENS, model.config.max_position_embeddings)
```

This replaces the previous behavior that set total `max_length` to the model's
full positional limit. The default `MAX_NEW_TOKENS=4096` matches the model's
document-oriented use while providing a finite ceiling. CPU smoke tests should
override it with a much smaller value.

The setting is read when generation arguments are built, so a validated config
update made before a request affects the existing client.

## 6. Two-stage page inference

TeleOCR is not a single OCR call. At a high level, each page undergoes:

1. **Layout analysis.** The page is normalized for layout inference and the
   model predicts regions and types.
2. **Block preparation.** Bounding boxes are validated, transformed from the
   model's normalized 0-1000 coordinate space, and cropped from the page.
3. **Content extraction.** Each eligible block is prompted according to its
   type, such as text, table, equation, code, seal, or scientific figure.
4. **Post-processing.** OCR text, LaTeX, table structure, and geometry are
   normalized into `ContentBlock` objects.
5. **Document conversion.** Blocks become middle JSON and reconstructed
   Markdown; a diagnostic layout PDF is drawn.

Consequences for capacity planning:

- a one-page document requires at least layout and content generation;
- pages with multiple blocks may require multiple content requests;
- layout inference can be slower than OCR of a small crop;
- `MAX_NEW_TOKENS` applies independently to every generation request;
- runtime does not scale only with source file size or page count.

## 7. PDF and image processing

Accepted images are normalized to PDF bytes before analysis. PDFs are rendered
through a lazy-selected backend:

- `PyMuPDF` loads `pdf_image_tools_PyMuPDF`;
- `pypdfium2` loads `pdf_image_tools_pdfium`.

The backend is selected at runtime through `CONFIG.PDF_TOOLS`. Lazy dispatch
prevents the unselected implementation from controlling behavior.

`PDF_TOOLS_WORKER_MAX_NUM=0` is recommended for constrained CPU validation. A
larger value enables PDF rendering workers but does not parallelize model
generation and can increase total memory pressure.

`MAX_PIXELS` rejects pages beyond the configured pixel budget. It is a resource
guard, not a model resolution setting.

## 8. Configuration semantics

Defaults:

| Key | Default | Validation and use |
| --- | ---: | --- |
| `model_path` | `StarDoc-AI/TeleOCR` | Must be a non-empty string. |
| `BACKEND` | `vllm-async-engine` | One of `transformers`, `vllm-engine`, `vllm-async-engine`. CPU must override it. |
| `LAYOUT_MODE` | `Detection` | `Detection` or `Segmentation`. |
| `MAX_MODEL_LEN` | `16384` | Positive; used by vLLM model construction. |
| `MAX_NEW_TOKENS` | `4096` | Positive; Transformers fallback generation ceiling. |
| `GPU_MEMORY_UTILIZATION` | `0.95` | In `(0, 1]`; used by vLLM. |
| `PDF_TOOLS` | `pypdfium2` | `PyMuPDF` or `pypdfium2`. |
| `PDF_TOOLS_WORKER_MAX_NUM` | `4` | Non-negative integer. |
| `PDF_TOOLS_WORKER_RATIO` | `0.7` | In `(0, 1]`. |
| `MAX_PIXELS` | `64000000` | Positive integer. |

`CONFIG.update()` builds a prospective configuration, validates the complete
set, and only then mutates globals. An invalid multi-key override therefore
does not partially change the active configuration.

CLI conversion follows each default's Python type. Keys are case-sensitive.
The literal `none` converts to `None` only for settings whose validation permits
it; current public settings generally require concrete values.

## 9. CLI execution and failure model

`infer.py` performs these preflight actions before inference:

- validates all overrides;
- creates the result root;
- verifies that the input path is a directory;
- selects top-level files, excluding `.json` and `.html`;
- sorts files by filename for deterministic execution;
- rejects duplicate filename stems.

Each document is isolated in a `try` block. A failed document is recorded and
the batch continues. Exit codes are:

- `0`: every document succeeded;
- `1`: at least one document failed;
- `2`: configuration or preflight failed.

`--use_async` is deprecated. Actual routing is determined by `BACKEND`.

## 10. Transactional output behavior

For each document, `_stage_output()` creates a hidden temporary directory under
the selected output root. All artifacts are written there. On success,
`_commit_output()` performs an atomic rename into the final document directory.

When overwriting an existing result:

1. Move the existing output to a uniquely named backup.
2. Move the completed staging directory to the final name.
3. Delete the backup only after the new result is in place.
4. Restore the backup if the replacement move fails.

On analysis or output failure, the staging directory is removed. Output names
containing path separators or traversal components are rejected.

This protects completed results from partial files, failed inference, and
path-escape attempts.

## 11. Intermediate JSON contract

The root object contains:

```json
{
  "pdf_info": [],
  "_backend": "vlm",
  "_version_name": "1.0.0"
}
```

Each `pdf_info` entry represents one page and includes:

- `page_idx`: original zero-based source page number;
- `page_size`: width and height;
- `para_blocks`: retained content blocks;
- `discarded_blocks`: regions excluded from the reconstructed content.

Content blocks contain geometry and type-specific data. Text-like blocks
typically include lines, spans, bounding boxes, types, and recognized content.

Page subsets are converted before inference, then mapped back to original page
indices. A request such as `[2, 0]` returns those pages in requested order while
preserving `page_idx` values `2` and `0`.

## 12. Library integration

Synchronous example:

```python
from pathlib import Path

import TeleOCR.config as CONFIG
from TeleOCR.engine import do_parse
from TeleOCR.tools.read_file import read_fn

CONFIG.update([
    "BACKEND=transformers",
    "MAX_NEW_TOKENS=256",
    "PDF_TOOLS=PyMuPDF",
    "PDF_TOOLS_WORKER_MAX_NUM=0",
])
CONFIG.validate()

path = Path("documents/report.pdf")
result = do_parse(
    output_dir="results",
    pdf_file_names=[path.stem],
    pdf_bytes_list=[read_fn(path)],
    valid_page_ids=[None],
    overwrite=False,
)
```

Contract requirements:

- `pdf_file_names`, `pdf_bytes_list`, and `valid_page_ids` must have matching
  lengths;
- each output name must be a plain filename component;
- each page selection is `None` or a non-empty unique list of integer indices;
- model/config initialization is process-global, so configure before serving
  requests and avoid changing global settings concurrently.

`aio_do_parse()` provides the asynchronous API. With Transformers, its model
work is delegated through thread-based async wrappers; it does not provide the
same scheduling semantics as the vLLM async engine.

## 13. Reproducible dependency model

`pyproject.toml` separates the package into:

- core document-processing dependencies;
- `transformers` inference dependencies;
- `vllm` GPU dependencies;
- optional server dependencies;
- development/test dependencies.

`requirements-cpu-lock.txt` pins the Python 3.12 Linux x86-64 CPU environment,
including hashes and CPU-only PyTorch artifacts. It must be installed with:

```bash
uv pip sync \
  --python .venv-cpu/bin/python \
  --torch-backend cpu \
  requirements-cpu-lock.txt
```

The editable package is then installed with `--no-deps` so resolution does not
silently replace locked packages.

The general CI matrix installs `.[dev]`, where optional PyTorch is absent. CPU
tests use `pytest.importorskip("torch")` in that environment. The dedicated
`cpu-lock` job installs the lock and runs the complete CPU-aware suite.

## 14. Tests and quality gates

The suite covers:

- atomic configuration updates and invalid values;
- CLI preflight, backend routing, and partial-failure exit status;
- page selection, ordering, validation, and source index preservation;
- output staging, overwrite recovery, and path containment;
- malformed model layout and post-processing failures;
- CPU model load defaults;
- bounded Transformers generation under inference mode;
- package/runtime version consistency.

Recommended pre-commit verification:

```bash
.venv-cpu/bin/python -m pytest -q
.venv-cpu/bin/ruff check .
.venv-cpu/bin/python -m compileall -q infer.py TeleOCR TeleOCR-vllm/TeleOCR_vllm
.venv-cpu/bin/python -m build
git diff --check
```

The GitHub Actions workflow runs Python 3.10 and 3.12 core jobs plus a Python
3.12 CPU-lock job.

## 15. Measured CPU behavior

The reference test used an Intel Core i7-7700 with four physical cores, eight
logical CPUs, and CPU-only PyTorch 2.8.0.

| Test | Limit | Elapsed | Peak RSS | Result |
| --- | ---: | ---: | ---: | --- |
| Direct text-image generation | 32 tokens | 36.15 s | 3.03 GB | Coherent OCR text. |
| Full CLI layout + extraction | 32 tokens/request | 9 min 34 s | 3.99 GB | Markdown, JSON, layout PDF; exit 0. |

The full run spent approximately four minutes in layout generation and five
minutes in extraction. These figures are an observed reference, not a service
level guarantee. Page dimensions, block count, CPU instruction support,
threading, memory bandwidth, and token limit all affect runtime.

## 16. Performance tuning

Safe tuning order:

1. Validate correctness with one page and `MAX_NEW_TOKENS=32`.
2. Set `OMP_NUM_THREADS` to the number of physical CPU cores allocated.
3. Keep `PDF_TOOLS_WORKER_MAX_NUM=0` until model memory is stable.
4. Increase `MAX_NEW_TOKENS` gradually based on observed truncation.
5. Process documents serially when RAM is constrained.
6. Consider a GPU/vLLM backend for sustained throughput.

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `OMP_NUM_THREADS` | Controls CPU math threading; TeleOCR defaults it to `1` if unset. |
| `MALLOC_ARENA_MAX=2` | Limits glibc allocator arenas and may reduce fragmentation. |
| `TOKENIZERS_PARALLELISM=false` | Avoids tokenizer thread-pool warnings and oversubscription. |
| `HF_MODULES_CACHE` | Redirects trusted model-code cache when the home directory is read-only. |

Avoid launching multiple model processes on a low-memory host. The model cache
on disk is shared, but each process has its own model and generation memory.

## 17. Security and trust boundaries

The checkpoint is loaded with `trust_remote_code=True`. This means model
repository Python code executes in the TeleOCR process. For controlled or
regulated environments:

- inspect the model repository code;
- pin a known model snapshot;
- use a local, read-only snapshot directory;
- limit network access after provisioning;
- run under an unprivileged account;
- treat input documents as untrusted data;
- retain the pixel and output-path guards;
- do not expose arbitrary config overrides directly to unauthenticated users.

Output atomicity protects filesystem consistency, not content confidentiality.
Input and recognized text may be sensitive and should follow the deployment's
storage, retention, and access-control policy.

## 18. Known limitations

- CPU latency is high, especially for layout analysis.
- The default model is downloaded from an external service on first use.
- The model's trusted remote code is part of the execution boundary.
- Global configuration and singleton model caching favor one stable
  configuration per process.
- The CLI scans one directory level and processes documents serially.
- `MAX_NEW_TOKENS` bounds generation but may truncate content.
- `MAX_MODEL_LEN` and `GPU_MEMORY_UTILIZATION` are vLLM settings and do not tune
  CPU Transformers inference.
- Increasing PDF workers can increase memory pressure without accelerating the
  model itself.

## 19. Maintenance procedure

When changing CPU inference:

1. Preserve the frozen baseline tag; add new commits instead of moving it.
2. Update `pyproject.toml` bounds intentionally.
3. Regenerate the CPU lock only when dependencies change.
4. Test in both the core-only environment and locked CPU environment.
5. Run a bounded direct generation test.
6. Run one complete CLI document and record elapsed time and peak RSS.
7. Confirm all output artifacts and JSON page mappings.
8. Push and require the GitHub CI workflow to pass.
9. Update both CPU guides if behavior or commands changed.

## 20. Diagnostic commands

Environment identity:

```bash
.venv-cpu/bin/python --version
.venv-cpu/bin/python -c \
  "import torch, transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available())"
```

Capacity:

```bash
free -h
df -h . ~/.cache/huggingface
```

Time and peak memory for a CLI run:

```bash
/usr/bin/time -v \
  .venv-cpu/bin/python infer.py \
    --image_sub_path cpu-input \
    --result_save_path cpu-output \
    --overwrite \
    --override \
      BACKEND=transformers \
      MAX_NEW_TOKENS=32 \
      PDF_TOOLS=PyMuPDF \
      PDF_TOOLS_WORKER_MAX_NUM=0
```

Repository state:

```bash
git status --short
git log -3 --oneline --decorate
```
