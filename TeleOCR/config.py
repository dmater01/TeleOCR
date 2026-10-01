# =========================
# Model
# =========================

model_path = "StarDoc-AI/TeleOCR"

BACKEND = "vllm-async-engine"
# [vllm-engine, vllm-async-engine]


# =========================
# Layout
# =========================

LAYOUT_MODE = "Detection"
# [Detection / Segmentation]


# =========================
# VLLM
# =========================

MAX_MODEL_LEN = 16384
MAX_NEW_TOKENS = 4096
GPU_MEMORY_UTILIZATION = 0.95


# =========================
# PDF
# =========================

PDF_TOOLS = "pypdfium2"
# [PyMuPDF / pypdfium2]

PDF_TOOLS_WORKER_MAX_NUM = 4
PDF_TOOLS_WORKER_RATIO = 0.7

MAX_PIXELS = 8000 * 8000

CONFIG_KEYS = {
    "model_path",
    "BACKEND",
    "LAYOUT_MODE",
    "MAX_MODEL_LEN",
    "MAX_NEW_TOKENS",
    "GPU_MEMORY_UTILIZATION",
    "PDF_TOOLS",
    "PDF_TOOLS_WORKER_MAX_NUM",
    "PDF_TOOLS_WORKER_RATIO",
    "MAX_PIXELS",
}


# =========================
# Runtime Override
# =========================

def _convert(value, reference):
    """根据默认值自动转换命令行参数类型。"""

    if isinstance(reference, bool):
        return value.lower() in {"true", "1", "yes"}

    if isinstance(reference, int):
        return int(value)

    if isinstance(reference, float):
        return float(value)

    if value.lower() == "none":
        return None

    return value


def _validate(values):
    if values["BACKEND"] not in {"transformers", "vllm-engine", "vllm-async-engine"}:
        raise ValueError(f"Unsupported BACKEND: {values['BACKEND']!r}")
    if values["LAYOUT_MODE"] not in {"Detection", "Segmentation"}:
        raise ValueError(f"Unsupported LAYOUT_MODE: {values['LAYOUT_MODE']!r}")
    if values["PDF_TOOLS"] not in {"PyMuPDF", "pypdfium2"}:
        raise ValueError(f"Unsupported PDF_TOOLS: {values['PDF_TOOLS']!r}")
    if values["MAX_MODEL_LEN"] <= 0:
        raise ValueError("MAX_MODEL_LEN must be positive")
    if values["MAX_NEW_TOKENS"] <= 0:
        raise ValueError("MAX_NEW_TOKENS must be positive")
    if not 0 < values["GPU_MEMORY_UTILIZATION"] <= 1:
        raise ValueError("GPU_MEMORY_UTILIZATION must be in (0, 1]")
    if values["PDF_TOOLS_WORKER_MAX_NUM"] < 0:
        raise ValueError("PDF_TOOLS_WORKER_MAX_NUM must be non-negative")
    if not 0 < values["PDF_TOOLS_WORKER_RATIO"] <= 1:
        raise ValueError("PDF_TOOLS_WORKER_RATIO must be in (0, 1]")
    if values["MAX_PIXELS"] <= 0:
        raise ValueError("MAX_PIXELS must be positive")
    if not isinstance(values["model_path"], str) or not values["model_path"].strip():
        raise ValueError("model_path must be a non-empty string")


def validate():
    """Validate the active configuration."""
    _validate({key: globals()[key] for key in CONFIG_KEYS})


def update(overrides):
    """Apply validated ``KEY=VALUE`` runtime overrides atomically."""
    changes = {}
    for item in overrides:
        key, sep, value = item.partition("=")

        if not sep:
            raise ValueError(
                f"Invalid config override: {item}. "
                f"Expected KEY=VALUE."
            )

        if key not in CONFIG_KEYS:
            raise KeyError(
                f"Unknown config option: {key}"
            )

        old_value = globals()[key]
        try:
            changes[key] = _convert(value, old_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {key}: {value!r}") from exc

    prospective = {key: globals()[key] for key in CONFIG_KEYS}
    prospective.update(changes)
    _validate(prospective)

    for key, new_value in changes.items():
        old_value = globals()[key]
        globals()[key] = new_value
        print(
            f"[Config] {key}: "
            f"{old_value!r} -> {new_value!r}"
        )


def show():
    """打印最终配置。"""

    print("\n========== TeleOCR Config ==========")

    for key, value in globals().items():

        if key.startswith("_"):
            continue

        if key in CONFIG_KEYS:
            print(f"{key} = {value}")

    print("====================================\n")
