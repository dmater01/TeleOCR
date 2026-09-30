import random
random.seed(42)
import argparse
import asyncio
from collections import Counter
from pathlib import Path

from TeleOCR.engine import aio_do_parse, do_parse
from TeleOCR.tools.read_file import read_fn
import TeleOCR.config as CONFIG


def parse_args():
    parser = argparse.ArgumentParser(
        description="TeleOCR batch inference"
    )

    parser.add_argument(
        "--image_sub_path",
        type=str,
        required=True,
        help="输入图片/PDF文件目录",
    )

    parser.add_argument(
        "--result_save_path",
        type=str,
        required=True,
        help="结果保存目录",
    )

    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="运行时覆盖 TeleOCR 配置",
    )

    parser.add_argument(
        "--use_async",
        action="store_true",
        help="Deprecated; execution mode is derived from BACKEND.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing per-document output directories.",
    )

    return parser.parse_args()

def get_image_paths(image_sub_path):
    if not image_sub_path.is_dir():
        raise ValueError(f"Input directory does not exist: {image_sub_path}")
    paths = [
        path
        for path in image_sub_path.iterdir()
        if path.is_file()
        and path.suffix.lower() not in {".json", ".html"}
    ]
    paths.sort(key=lambda path: path.name)
    duplicate_stems = [stem for stem, count in Counter(path.stem for path in paths).items() if count > 1]
    if duplicate_stems:
        raise ValueError(f"Input files have colliding stems: {', '.join(sorted(duplicate_stems))}")
    return paths

async def async_main(image_paths, result_save_path, overwrite=False):
    failures = []
    for pdf_path in image_paths:
        print(f"\nProcessing: {pdf_path}")
        try:
            data = read_fn(pdf_path)
            await aio_do_parse(
                str(result_save_path),
                [pdf_path.stem],
                [data],
                valid_page_ids=[None],
                overwrite=overwrite,
            )
            print(
                f"Finished: {pdf_path.name}"
            )
        except Exception as e:
            failures.append((pdf_path, e))
            print(
                f"Failed: {pdf_path.name}\n"
                f"Error: {e}"
            )
    return failures

def sync_main(image_paths, result_save_path, overwrite=False):
    failures = []
    for pdf_path in image_paths:

        print(f"\nProcessing: {pdf_path}")

        try:
            data = read_fn(pdf_path)

            do_parse(
                str(result_save_path),
                [pdf_path.stem],
                [data],
                valid_page_ids=[None],
                overwrite=overwrite,
            )
            print(
                f"Finished: {pdf_path.name}"
            )

        except Exception as e:
            failures.append((pdf_path, e))

            print(
                f"Failed: {pdf_path.name}\n"
                f"Error: {e}"
            )
    return failures

def main():

    args = parse_args()

    try:
        CONFIG.update(args.override)
        CONFIG.validate()
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return 2
    CONFIG.show()

    if args.use_async:
        print("Warning: --use_async is deprecated; BACKEND determines execution mode.")


    image_sub_path = Path(
        args.image_sub_path
    )

    result_save_path = Path(
        args.result_save_path
    )

    try:
        result_save_path.mkdir(parents=True, exist_ok=True)
        image_paths = get_image_paths(image_sub_path)
    except (OSError, ValueError) as exc:
        print(f"Preflight error: {exc}")
        return 2

    print(
        f"Found {len(image_paths)} files."
    )

    if CONFIG.BACKEND == "vllm-async-engine":
        failures = asyncio.run(
            async_main(
                image_paths,
                result_save_path,
                args.overwrite,
            )
        )
    else:
        failures = sync_main(
            image_paths,
            result_save_path,
            args.overwrite,
        )

    print(f"Completed: {len(image_paths) - len(failures)} succeeded, {len(failures)} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
