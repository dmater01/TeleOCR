import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from loguru import logger

from TeleOCR.tools.pdf_image_tools import convert_pdf_bytes_to_bytes
from TeleOCR.data_reader_writer import FileBasedDataWriter, ImageDataWriter
from TeleOCR.tools.draw_bbox import draw_layout_bbox
from TeleOCR.src.vlm_middle_json_mkcontent import union_make
from TeleOCR.src.vlm_analyze import doc_analyze
from TeleOCR.src.vlm_analyze import aio_doc_analyze 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def prepare_env(local_md_dir):
    local_md_dir = str(local_md_dir)
    local_image_dir = os.path.join(str(local_md_dir), "images")
    os.makedirs(local_image_dir, exist_ok=True)
    return local_image_dir, local_md_dir


def _stage_output(output_dir, pdf_file_name, overwrite):
    if Path(pdf_file_name).name != pdf_file_name or pdf_file_name in {"", ".", ".."}:
        raise ValueError(f"Invalid output name: {pdf_file_name!r}")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / pdf_file_name
    if final_dir.exists():
        nonempty = final_dir.is_file() or any(final_dir.iterdir())
        if nonempty and not overwrite:
            raise FileExistsError(f"Output already exists: {final_dir}; use overwrite=True")
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{pdf_file_name}-", dir=output_root))
    return stage_dir, final_dir


def _commit_output(stage_dir, final_dir):
    backup = None
    if final_dir.exists():
        backup = final_dir.with_name(f".{final_dir.name}.backup-{uuid.uuid4().hex}")
        os.replace(final_dir, backup)
    try:
        os.replace(stage_dir, final_dir)
    except Exception:
        if backup is not None:
            os.replace(backup, final_dir)
        raise
    if backup is not None:
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def _prepare_pdf_bytes(pdf_bytes_list, valid_page_ids):
    if len(pdf_bytes_list) != len(valid_page_ids):
        raise ValueError("pdf_bytes_list and valid_page_ids must have the same length")

    prepared = []
    source_page_maps = []
    for pdf_bytes, page_ids in zip(pdf_bytes_list, valid_page_ids):
        if page_ids is not None:
            if not page_ids:
                raise ValueError("valid_page_ids entries must be None or a non-empty list")
            if any(not isinstance(page_id, int) or isinstance(page_id, bool) for page_id in page_ids):
                raise ValueError("page IDs must be integers")
            if len(page_ids) != len(set(page_ids)):
                raise ValueError("page IDs must not contain duplicates")
            page_ids = list(page_ids)
        prepared.append(convert_pdf_bytes_to_bytes(pdf_bytes, page_ids))
        source_page_maps.append(page_ids)
    return prepared, source_page_maps


def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        middle_json,
        f_draw_layout_bbox=True,
        f_dump_md=True,
        f_dump_middle_json=True,
):
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
    image_dir = str(os.path.basename(local_image_dir))
    if f_dump_md:
        md_content_str = union_make(pdf_info, image_dir) 
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )
    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )
    logger.info(f"local output dir is {local_md_dir}")

async def _async_process_vlm(
    output_dir, pdf_file_names, pdf_bytes_list, source_page_maps, overwrite=False, **kwargs
):
    results = []
    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[idx]
        stage_dir, final_dir = _stage_output(output_dir, pdf_file_name, overwrite)
        try:
            local_image_dir, local_md_dir = prepare_env(stage_dir)
            image_writer = ImageDataWriter(local_image_dir)
            vlm_doc_analyze_time = time.time()
            middle_json = await aio_doc_analyze(pdf_bytes, image_writer=image_writer, **kwargs)
            logger.debug(f"doc_analyze cost: {round(time.time() - vlm_doc_analyze_time, 2)}")
            inplace_change_page_ids([middle_json], [source_page_maps[idx]])
            pdf_info = middle_json["pdf_info"]
            md_writer = FileBasedDataWriter(local_md_dir)
            process_output_time = time.time()
            _process_output(
                pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
                md_writer, middle_json
            )
            image_writer.save_all_images()
            _commit_output(stage_dir, final_dir)
            logger.debug(f"process_output_time cost: {round(time.time() - process_output_time, 2)}")
            results.append(middle_json)
        except Exception:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
    return results


def _process_vlm(
    output_dir, pdf_file_names, pdf_bytes_list, source_page_maps, overwrite=False, **kwargs
):
    results = []
    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[idx]
        stage_dir, final_dir = _stage_output(output_dir, pdf_file_name, overwrite)
        try:
            local_image_dir, local_md_dir = prepare_env(stage_dir)
            image_writer = ImageDataWriter(local_image_dir)
            vlm_doc_analyze_time = time.time()
            middle_json = doc_analyze(pdf_bytes, image_writer=image_writer, **kwargs)
            logger.debug(f"doc_analyze cost: {round(time.time() - vlm_doc_analyze_time, 2)}")
            inplace_change_page_ids([middle_json], [source_page_maps[idx]])
            pdf_info = middle_json["pdf_info"]
            md_writer = FileBasedDataWriter(local_md_dir)
            process_output_time = time.time()
            _process_output(
                pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
                md_writer, middle_json
            )
            image_writer.save_all_images()
            _commit_output(stage_dir, final_dir)
            logger.debug(f"process_output_time cost: {round(time.time() - process_output_time, 2)}")
            results.append(middle_json)
        except Exception:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
    return results


def inplace_change_page_ids(results, valid_page_ids):
    for result, source_page_ids in zip(results, valid_page_ids):
        if source_page_ids is None:
            continue
        for page in result.get("pdf_info", []):
            page_id = page.get("page_idx")
            if (
                isinstance(page_id, int)
                and 0 <= page_id < len(source_page_ids)
            ):
                page["page_idx"] = source_page_ids[page_id]
    return results


async def aio_do_parse( 
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        valid_page_ids: list[list[int] | None],
        **kwargs,
):
    if len(pdf_file_names) != len(pdf_bytes_list):
        raise ValueError("pdf_file_names and pdf_bytes_list must have the same length")
    pdf_bytes_list, source_page_maps = _prepare_pdf_bytes(pdf_bytes_list, valid_page_ids)
    results = await _async_process_vlm(
        output_dir, pdf_file_names, pdf_bytes_list, source_page_maps, **kwargs,
    )
    return results


def do_parse(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        valid_page_ids: list[list[int] | None],
        **kwargs,
):
    if len(pdf_file_names) != len(pdf_bytes_list):
        raise ValueError("pdf_file_names and pdf_bytes_list must have the same length")
    pdf_bytes_list, source_page_maps = _prepare_pdf_bytes(pdf_bytes_list, valid_page_ids)
    results = _process_vlm(
        output_dir, pdf_file_names, pdf_bytes_list, source_page_maps, **kwargs,
    )
    return results
