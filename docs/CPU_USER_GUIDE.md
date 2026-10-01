# TeleOCR CPU User Guide

This guide explains how to install, run, verify, and troubleshoot TeleOCR on a
Linux CPU-only system. It targets the hardened CPU path on the `main` branch of
[`dmater01/TeleOCR`](https://github.com/dmater01/TeleOCR).

For implementation details, see the [CPU Technical Guide](CPU_TECHNICAL_GUIDE.md).

## 1. What the CPU version provides

The CPU path uses the Hugging Face Transformers backend. It:

- runs without CUDA or vLLM;
- loads the checkpoint in its native dtype instead of expanding it to float32;
- enables low-memory model loading;
- runs generation under PyTorch inference mode;
- limits generated output with `MAX_NEW_TOKENS`;
- writes Markdown, structured JSON, and an annotated layout PDF;
- stages output and replaces an existing result atomically when `--overwrite`
  is used.

CPU inference is functional but slow. It is intended for validation,
development, occasional documents, and low-volume workloads. A GPU remains the
better choice for interactive or high-throughput OCR.

## 2. Tested baseline

The CPU implementation was validated with:

- Linux x86-64;
- Python 3.12;
- PyTorch 2.8.0 CPU;
- Transformers 4.57.6;
- a 4-core/8-thread Intel Core i7-7700;
- the public `StarDoc-AI/TeleOCR` checkpoint.

The full two-stage test processed one image in 9 minutes 34 seconds, peaked at
3.99 GB resident memory, and produced all three output artifacts. The short
test used `MAX_NEW_TOKENS=32`, so its text was intentionally bounded.

Recommended free capacity before installation:

| Resource | Minimum practical target | Notes |
| --- | ---: | --- |
| RAM available to TeleOCR | 5 GB | More headroom is recommended for large pages. |
| Disk space | 6 GB | Covers the model cache and CPU environment. |
| Swap | 2 GB | A safety margin, not a substitute for RAM. |
| CPU | x86-64, 4 cores | More cores can improve inference time. |

## 3. Get the CPU-enabled code

For a new checkout:

```bash
git clone https://github.com/dmater01/TeleOCR.git
cd TeleOCR
git checkout main
```

For an existing checkout with the `fork` remote already configured:

```bash
git switch main
git pull --ff-only fork main
```

Confirm that the CPU changes are present:

```bash
git log -2 --oneline
```

The history should include:

```text
28a08b4 Skip optional CPU tests without PyTorch
f189195 Make Transformers inference safe on CPU
```

The tag `cpu-test-baseline-1.0.0` is the earlier frozen hardening baseline. Use
`main`, not that tag, when you want the later CPU inference improvements.

## 4. Install the reproducible CPU environment

The lock file is for Python 3.12 on Linux x86-64. Check the prerequisites:

```bash
uv --version
uv python find 3.12
```

From the repository root:

```bash
uv venv .venv-cpu --python 3.12

uv pip sync \
  --python .venv-cpu/bin/python \
  --torch-backend cpu \
  requirements-cpu-lock.txt

uv pip install \
  --python .venv-cpu/bin/python \
  --no-deps \
  -e .
```

The editable install makes source changes immediately visible without
reinstalling the project.

Verify the CPU runtime:

```bash
.venv-cpu/bin/python -c \
  "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected characteristics:

```text
2.8.0+cpu
CUDA: False
```

Run the automated checks:

```bash
.venv-cpu/bin/python -m pytest -q
.venv-cpu/bin/ruff check .
```

The validated release reports 33 passing tests in the locked environment.

## 5. Prepare input

Create dedicated input and output directories:

```bash
mkdir -p cpu-input cpu-output
```

Place files directly in `cpu-input`. Supported formats are:

- PDF;
- PNG;
- JPEG/JPG;
- JPEG 2000 (`.jp2`);
- WebP;
- GIF;
- BMP;
- TIFF.

The CLI processes files in the top level of the input directory. It does not
recursively scan subdirectories. File type is checked from content, not only
the filename extension.

Do not place two files with the same stem in one batch. For example,
`invoice.pdf` and `invoice.png` would both target `cpu-output/invoice` and are
therefore rejected during preflight.

## 6. Run a smoke test

Start with one short page and a 32-token ceiling:

```bash
OMP_NUM_THREADS=4 \
MALLOC_ARENA_MAX=2 \
TOKENIZERS_PARALLELISM=false \
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

On the first run, Transformers downloads the approximately 2.8 GB model into
the Hugging Face cache. Later runs reuse it.

`OMP_NUM_THREADS=4` is a sensible starting point for a four-core CPU. Adjust it
to the number of physical cores available to the process. If it is omitted,
TeleOCR defaults it to `1` before loading the model.

## 7. Run normal CPU OCR

After the smoke test passes, increase the output limit:

```bash
OMP_NUM_THREADS=4 \
MALLOC_ARENA_MAX=2 \
TOKENIZERS_PARALLELISM=false \
.venv-cpu/bin/python infer.py \
  --image_sub_path cpu-input \
  --result_save_path cpu-output \
  --overwrite \
  --override \
    BACKEND=transformers \
    MAX_NEW_TOKENS=512 \
    PDF_TOOLS=PyMuPDF \
    PDF_TOOLS_WORKER_MAX_NUM=0
```

Token-limit guidance:

| `MAX_NEW_TOKENS` | Suggested use | Tradeoff |
| ---: | --- | --- |
| 32 | Installation smoke test | Fastest, but output is usually truncated. |
| 128 | Short blocks or quick validation | May truncate dense text or tables. |
| 256 | Ordinary short pages | Moderate CPU runtime. |
| 512 | Longer pages | Slower and uses a larger generation cache. |
| 4096 | Project default | Highest completion allowance; potentially very slow on CPU. |

The limit applies to each Transformers generation request. TeleOCR normally
performs at least two stages per page: layout analysis and content extraction.
A page containing several detected blocks can require additional extraction
requests.

## 8. Use a local model directory

By default, `model_path=StarDoc-AI/TeleOCR` resolves through Hugging Face. For
offline operation, first place a complete model snapshot on local storage and
then pass its directory:

```bash
.venv-cpu/bin/python infer.py \
  --image_sub_path cpu-input \
  --result_save_path cpu-output \
  --override \
    BACKEND=transformers \
    model_path=/absolute/path/to/TeleOCR-model \
    MAX_NEW_TOKENS=256 \
    PDF_TOOLS=PyMuPDF \
    PDF_TOOLS_WORKER_MAX_NUM=0
```

The model uses Hugging Face trusted remote code. In controlled deployments,
review the model repository and pin a known snapshot instead of relying on a
moving remote revision.

## 9. Understand the output

For an input named `invoice.pdf`, the result is:

```text
cpu-output/
└── invoice/
    ├── images/
    ├── invoice.md
    ├── invoice_middle.json
    └── invoice_layout.pdf
```

- `invoice.md` is the reconstructed document content.
- `invoice_middle.json` contains page geometry, detected blocks, lines, spans,
  recognized content, backend metadata, and source page indices.
- `invoice_layout.pdf` overlays the detected layout on the original pages.
- `images/` contains extracted image assets when the document produces them.

Outputs are built in a hidden staging directory. A successful run atomically
moves the completed directory into place. If processing fails, TeleOCR removes
the staging directory and leaves an existing completed result intact.

Without `--overwrite`, a non-empty existing result directory is protected and
the document fails with an instruction to enable overwrite.

## 10. CLI behavior and exit codes

Show the command help:

```bash
.venv-cpu/bin/python infer.py --help
```

Main arguments:

| Argument | Required | Meaning |
| --- | --- | --- |
| `--image_sub_path DIR` | Yes | Directory containing input documents. |
| `--result_save_path DIR` | Yes | Root directory for results. |
| `--override KEY=VALUE ...` | No | Validated runtime configuration changes. |
| `--overwrite` | No | Atomically replace existing per-document results. |
| `--use_async` | No | Deprecated; execution mode follows `BACKEND`. |

Exit status:

| Code | Meaning |
| ---: | --- |
| 0 | Every input succeeded. |
| 1 | One or more documents failed. |
| 2 | Configuration, argument, or preflight failure. |

The batch continues after an individual document failure and prints a final
success/failure count.

## 11. CPU configuration reference

| Setting | CPU recommendation | Description |
| --- | --- | --- |
| `BACKEND` | `transformers` | Required for this CPU environment. |
| `model_path` | `StarDoc-AI/TeleOCR` or local path | Model identifier or snapshot directory. |
| `LAYOUT_MODE` | `Detection` | `Detection` or `Segmentation`. |
| `MAX_NEW_TOKENS` | Start at `32`; raise as needed | Maximum generated tokens per request. |
| `PDF_TOOLS` | `PyMuPDF` | PDF rasterization backend; `pypdfium2` is also supported. |
| `PDF_TOOLS_WORKER_MAX_NUM` | `0` | Avoids extra PDF workers during constrained CPU tests. |
| `PDF_TOOLS_WORKER_RATIO` | `0.7` | Worker resource ratio when workers are enabled. |
| `MAX_PIXELS` | `64000000` | Maximum allowed pixels for a processed PDF page. |
| `MAX_MODEL_LEN` | Leave at `16384` | vLLM context setting; not the CPU generation limit. |
| `GPU_MEMORY_UTILIZATION` | Leave at `0.95` | vLLM GPU setting; unused by CPU Transformers. |

All override keys are case-sensitive. Invalid keys and invalid values are
rejected before input processing starts.

## 12. Page selection from Python

The CLI processes all pages. Library callers can select and reorder pages with
`valid_page_ids`:

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

source = Path("cpu-input/report.pdf")
results = do_parse(
    "cpu-output",
    [source.stem],
    [read_fn(source)],
    valid_page_ids=[[2, 0]],
    overwrite=True,
)
```

Page numbers are zero-based. `None` selects every page. A selection must be a
non-empty list of unique integers in range. Returned `page_idx` values retain
the original source page numbers, including custom order.

## 13. Troubleshooting

### `No module named 'torch'` or `No module named 'transformers'`

The core package is installed without optional ML frameworks. Re-run the CPU
lock installation and use `.venv-cpu/bin/python`, not the system interpreter.

### CUDA appears enabled

Confirm that the lock was installed with `--torch-backend cpu`:

```bash
.venv-cpu/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The version should end with `+cpu`, and CUDA should be `False`.

### The process is killed or the system becomes unresponsive

- Close memory-heavy applications.
- Process one document at a time.
- Use `PDF_TOOLS_WORKER_MAX_NUM=0`.
- Start with `MAX_NEW_TOKENS=32`.
- Ensure at least 5 GB of memory is available before model loading.
- Add swap if the machine has little RAM, while expecting slower execution.

### OCR output stops mid-sentence

The generation ceiling was reached. Increase `MAX_NEW_TOKENS` gradually. The
32-token smoke setting is specifically expected to truncate longer content.

### Processing seems frozen at `Predict: 0%`

CPU generation may produce no progress update until a complete request ends.
On the tested i7-7700, the layout stage alone took about four minutes. Check CPU
utilization before terminating the process.

### `Output already exists`

Choose a new output directory or add `--overwrite`. Overwrite uses staged,
atomic replacement so a failed new run does not destroy the old result.

### `Input files have colliding stems`

Rename or separate files such as `invoice.pdf` and `invoice.png` so each input
has a unique filename stem.

### Model download or authentication errors

Verify internet access and available disk space. The public model normally
works without authentication, although authenticated Hugging Face requests can
have higher rate limits. For offline use, pass a complete local model path.

### Trusted model code cannot create its cache

If the home cache is read-only, redirect generated model modules to a writable
directory:

```bash
HF_MODULES_CACHE=/tmp/teleocr-hf-modules \
.venv-cpu/bin/python infer.py ...
```

### PDF backend error

Switch between:

```text
PDF_TOOLS=PyMuPDF
PDF_TOOLS=pypdfium2
```

Both are included in the locked environment.

## 14. Updating and cleaning up

Update the code without changing the environment:

```bash
git pull --ff-only origin main
```

If `pyproject.toml` or `requirements-cpu-lock.txt` changes, synchronize again:

```bash
uv pip sync \
  --python .venv-cpu/bin/python \
  --torch-backend cpu \
  requirements-cpu-lock.txt
uv pip install --python .venv-cpu/bin/python --no-deps -e .
```

The virtual environment can be deleted and recreated from the lock file. The
downloaded model is stored in the Hugging Face cache and can also be removed if
disk space is needed; the next online run downloads it again.

## 15. Operational checklist

Before a CPU run:

- Confirm `git status` contains no unintended changes.
- Confirm the interpreter is `.venv-cpu/bin/python`.
- Confirm `torch.cuda.is_available()` is `False`.
- Confirm enough RAM, swap, and disk are available.
- Use unique input filename stems.
- Start with a small token limit for unfamiliar documents.
- Preserve previous output or use `--overwrite` intentionally.
- Expect several minutes per page on older desktop CPUs.
