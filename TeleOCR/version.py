from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("TeleOCR")
except PackageNotFoundError:
    __version__ = "1.0.0"
