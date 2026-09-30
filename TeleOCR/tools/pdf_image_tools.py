"""Lazy dispatch for the configured PDF implementation."""

import importlib
from functools import lru_cache

import TeleOCR.config as CONFIG


PDF_MODULES = {
    "PyMuPDF": ".pdf_image_tools_PyMuPDF",
    "pypdfium2": ".pdf_image_tools_pdfium",
}


@lru_cache(maxsize=len(PDF_MODULES))
def _load_backend(name: str):
    try:
        module_name = PDF_MODULES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported PDF_TOOLS value: {name!r}") from exc
    return importlib.import_module(module_name, package=__package__)


def _backend():
    return _load_backend(CONFIG.PDF_TOOLS)


def pdf_page_to_image(*args, **kwargs):
    return _backend().pdf_page_to_image(*args, **kwargs)


def _load_images_from_pdf_worker(*args, **kwargs):
    return _backend()._load_images_from_pdf_worker(*args, **kwargs)


def load_images_from_pdf(*args, **kwargs):
    return _backend().load_images_from_pdf(*args, **kwargs)


def load_images_from_pdf_core(*args, **kwargs):
    return _backend().load_images_from_pdf_core(*args, **kwargs)


def cut_image(*args, **kwargs):
    return _backend().cut_image(*args, **kwargs)


def get_crop_img(*args, **kwargs):
    return _backend().get_crop_img(*args, **kwargs)


def images_bytes_to_pdf_bytes(*args, **kwargs):
    return _backend().images_bytes_to_pdf_bytes(*args, **kwargs)


def get_page_size(*args, **kwargs):
    return _backend().get_page_size(*args, **kwargs)


def convert_pdf_bytes_to_bytes(*args, **kwargs):
    return _backend().convert_pdf_bytes_to_bytes(*args, **kwargs)
