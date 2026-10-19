"""OCR extraction for scanned PDFs using vision LLMs."""

import asyncio
import hashlib
import io
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vector_core import file_lock
from vector_core.settings import settings as vector_settings

from mcp_docs.extraction.vision_client import (
    VisionOCRClient,
    VisionOCRError,
    VisionOCRUnavailableError,
)
from mcp_docs.models import ExtractedContent, ExtractionError
from mcp_docs.settings import settings

logger = logging.getLogger(__name__)


class IncompleteOCRError(ExtractionError):
    """OCR left pages that a later attempt may still complete.

    Completed pages are kept as progress, so the next attempt only sends the
    pages listed in ``failed_pages`` to the vision endpoint. If the progress
    could not be saved, ``progress_error`` holds the reason and the next
    attempt starts from the last progress that was saved.
    """

    def __init__(
        self,
        filename: str,
        failed_pages: list[int],
        page_count: int,
        progress_error: OSError | None = None,
    ) -> None:
        self.failed_pages = failed_pages
        self.page_count = page_count
        self.progress_error = progress_error
        outcome = (
            "Completed pages are kept."
            if progress_error is None
            else f"Progress could not be saved ({progress_error}), so the next attempt "
            "starts from the last saved progress."
        )
        super().__init__(
            f"OCR incomplete for {filename}: {len(failed_pages)} of {page_count} pages "
            f"failed and will be retried ({_describe_pages(failed_pages)}). {outcome}"
        )


def _describe_pages(pages: list[int]) -> str:
    """Compact page list, e.g. "page 3" or "pages 1, 4, 10-12"."""
    ranges: list[str] = []
    for first, last in _page_runs(pages, batch_size=len(pages) or 1):
        ranges.append(str(first) if first == last else f"{first}-{last}")
    return ("page " if len(pages) == 1 else "pages ") + ", ".join(ranges)


def _failed_placeholder(page: int) -> str:
    return f"[OCR failed for page {page}]"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _get_cache_dir() -> Path:
    """Get the OCR cache directory, creating if needed."""
    cache_dir = vector_settings.shared_data_dir / "ocr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_key(path: Path) -> str:
    """
    Compute a fast cache key from file metadata (path + size + mtime).

    Much faster than hashing file contents for large files.
    """
    stat = path.stat()
    key_data = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def _get_cache_path(cache_key: str) -> Path:
    """Get the cache file path for a cache key."""
    return _get_cache_dir() / f"{cache_key}.json"


def _load_from_cache(cache_key: str) -> ExtractedContent | None:
    """Load OCR result from cache if available."""
    cache_path = _get_cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path) as f:
            data = json.load(f)

        logger.info(f"Loaded OCR result from cache: {cache_key[:12]}...")
        return ExtractedContent(
            text=data["text"],
            title=data.get("title"),
            page_count=data.get("page_count", 0),
            word_count=data.get("word_count", 0),
            metadata=data.get("metadata", {}),
        )
    except Exception as e:
        logger.warning(f"Failed to load OCR cache {cache_key}: {e}")
        return None


def _save_to_cache(cache_key: str, result: ExtractedContent) -> None:
    """Save OCR result to cache."""
    cache_path = _get_cache_path(cache_key)
    try:
        data = {
            "text": result.text,
            "title": result.title,
            "page_count": result.page_count,
            "word_count": result.word_count,
            "metadata": result.metadata,
        }
        _write_json_atomic(cache_path, data)
        logger.debug(f"Saved OCR result to cache: {cache_key[:12]}...")
    except Exception as e:
        logger.warning(f"Failed to save OCR cache {cache_key}: {e}")


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON so concurrent readers never see a partial file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Per-page progress
# ---------------------------------------------------------------------------

_PROGRESS_VERSION = 1


@dataclass
class _OCRProgress:
    """Per-page OCR state of a document whose OCR has not completed."""

    page_count: int
    pages: dict[int, str] = field(default_factory=dict)  # completed page text
    attempts: dict[int, int] = field(default_factory=dict)  # failures charged to a page

    def pending(self, max_attempts: int) -> list[int]:
        """Pages that are neither complete nor out of attempts."""
        return [
            page
            for page in range(1, self.page_count + 1)
            if page not in self.pages and self.attempts.get(page, 0) < max_attempts
        ]

    def record(self, page: int, outcome: str | VisionOCRError) -> None:
        if isinstance(outcome, str):
            self.pages[page] = outcome
            self.attempts.pop(page, None)
        elif not isinstance(outcome, VisionOCRUnavailableError):
            # An outage says nothing about the page, so only other failures
            # use up its attempts.
            self.attempts[page] = self.attempts.get(page, 0) + 1

    def to_content(self, *, exhausted: bool) -> ExtractedContent:
        """Merge pages, with a placeholder for each page that has no text."""
        page_numbers = range(1, self.page_count + 1)
        failed_pages = [page for page in page_numbers if page not in self.pages]
        full_text = _merge_pages(
            [self.pages.get(page, _failed_placeholder(page)) for page in page_numbers]
        )
        word_count = len(full_text.split()) if full_text else 0

        metadata: dict[str, Any] = {
            "ocr_method": "vision_llm",
            "ocr_dpi": settings.ocr_dpi,
        }
        if settings.ocr_vision_model:
            metadata["ocr_model"] = settings.ocr_vision_model
        if failed_pages:
            metadata["ocr_failed_pages"] = failed_pages
            if exhausted:
                metadata["ocr_attempts_exhausted"] = True

        logger.info(
            f"OCR complete: {word_count} words from {self.page_count} pages "
            f"({len(failed_pages)} failed)"
        )
        return ExtractedContent(
            text=full_text,
            title=None,
            page_count=self.page_count,
            word_count=word_count,
            metadata=metadata,
        )


def _get_progress_path(cache_key: str) -> Path:
    return _get_cache_dir() / f"{cache_key}.progress.json"


def _load_progress(cache_key: str, page_count: int) -> _OCRProgress | None:
    """Load saved progress for this file, if it matches the current page count."""
    path = _get_progress_path(cache_key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("version") != _PROGRESS_VERSION or data.get("page_count") != page_count:
            return None
        pages = {int(page): text for page, text in data["pages"].items()}
        attempts = {int(page): int(count) for page, count in data["attempts"].items()}
    except Exception as e:
        logger.warning(f"Ignoring unreadable OCR progress {cache_key[:12]}...: {e}")
        return None
    if not all(isinstance(text, str) for text in pages.values()) or not all(
        1 <= page <= page_count for page in (*pages, *attempts)
    ):
        logger.warning(f"Ignoring inconsistent OCR progress {cache_key[:12]}...")
        return None
    return _OCRProgress(page_count, pages, attempts)


def _save_progress(cache_key: str, progress: _OCRProgress) -> None:
    """Save progress for the next attempt.

    Raises:
        OSError: If the progress file cannot be written
    """
    _write_json_atomic(
        _get_progress_path(cache_key),
        {
            "version": _PROGRESS_VERSION,
            "page_count": progress.page_count,
            "pages": {str(page): text for page, text in sorted(progress.pages.items())},
            "attempts": {str(page): n for page, n in sorted(progress.attempts.items())},
        },
    )


def _remove_progress(cache_key: str) -> None:
    try:
        _get_progress_path(cache_key).unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"Failed to remove OCR progress {cache_key[:12]}...: {e}")


_PAGE_HEADER = re.compile(r"^--- Page (\d+) ---\n\n", re.MULTILINE)


def _progress_from_legacy(cached: ExtractedContent | None, page_count: int) -> _OCRProgress | None:
    """Recover completed pages from a partial result cached by an older version.

    Older versions cached results with failed pages. Their text is the
    ``_merge_pages`` output, so completed pages can be split back out. Anything
    that does not parse exactly returns None, and every page is OCR'd again.
    """
    if cached is None or cached.page_count != page_count:
        return None
    failed = cached.metadata.get("ocr_failed_pages")
    if not isinstance(failed, list) or not all(isinstance(page, int) for page in failed):
        return None
    failed_pages = set(failed)
    if not failed_pages or not failed_pages <= set(range(1, page_count + 1)):
        return None
    if page_count == 1:
        # A single page is stored without a header, and that page failed.
        return _OCRProgress(page_count)

    text = cached.text
    headers = list(_PAGE_HEADER.finditer(text))
    numbers = [int(header.group(1)) for header in headers]
    if (
        not headers
        or headers[0].start() != 0
        or numbers != sorted(set(numbers))
        or numbers[-1] > page_count
    ):
        return None

    bodies: dict[int, str] = {}
    for index, header in enumerate(headers):
        if index + 1 < len(headers):
            body = text[header.end() : headers[index + 1].start()]
            if not body.endswith("\n\n"):
                return None
            body = body[:-2]
        else:
            body = text[header.end() :]
        bodies[numbers[index]] = body

    progress = _OCRProgress(page_count)
    for page in range(1, page_count + 1):
        if page in failed_pages:
            if bodies.get(page) != _failed_placeholder(page):
                return None
        else:
            # _merge_pages omits a page whose OCR text was empty.
            progress.pages[page] = bodies.get(page, "")
    return progress


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------

def _prepare_image(img: Any, max_dim: int, use_jpeg: bool, jpeg_quality: int) -> bytes:
    """Resize image if needed and convert to compressed bytes. Lazy-imports PIL."""
    from PIL import Image as PILImage

    width, height = img.size
    if width > max_dim or height > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        logger.debug(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

    buffer = io.BytesIO()
    if use_jpeg:
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=jpeg_quality)
    else:
        img.save(buffer, format="PNG", optimize=True)

    return buffer.getvalue()


def _pdf2image() -> Any:
    try:
        import pdf2image
    except ImportError as e:
        raise ExtractionError(
            "pdf2image not installed. Install with: pip install mcp-docs[ocr]\n"
            "Also requires poppler: apt install poppler-utils (Linux) "
            "or brew install poppler (macOS)"
        ) from e
    return pdf2image


def _count_pages(path: Path) -> int:
    """Number of pages to OCR: the PDF's page count, capped at ocr_max_pages.

    Raises:
        ExtractionError: If the PDF cannot be read or has no pages
    """
    pdf2image = _pdf2image()
    try:
        total_pages = pdf2image.pdfinfo_from_path(str(path))["Pages"]
    except Exception as e:
        raise ExtractionError(f"Failed to read PDF info: {e}") from e

    if total_pages == 0:
        raise ExtractionError(f"No pages found in PDF: {path.name}")

    if total_pages > settings.ocr_max_pages:
        logger.warning(
            f"PDF has {total_pages} pages, limiting OCR to first {settings.ocr_max_pages}"
        )
    return int(min(total_pages, settings.ocr_max_pages))


def _page_runs(pages: Iterable[int], batch_size: int) -> Iterator[tuple[int, int]]:
    """Group sorted page numbers into consecutive runs of at most batch_size pages."""
    run: list[int] = []
    for page in pages:
        if run and (page != run[-1] + 1 or len(run) >= batch_size):
            yield run[0], run[-1]
            run = []
        run.append(page)
    if run:
        yield run[0], run[-1]


def _render_pages(path: Path, pages: Iterable[int], batch_size: int = 20) -> dict[int, bytes]:
    """
    Render the given 1-based pages to compressed image bytes.

    Renders in batches to limit peak memory: only one batch of raw PIL images
    is held at a time, and the returned JPEG/PNG bytes are much smaller.
    A page that renders to no image is left out of the result.
    """
    convert_from_path = _pdf2image().convert_from_path
    use_jpeg = settings.ocr_image_format.lower() == "jpeg"
    max_dim = settings.ocr_image_max_dimension
    jpeg_quality = settings.ocr_jpeg_quality
    images: dict[int, bytes] = {}

    def render(first: int, last: int) -> list[Any]:
        return list(
            convert_from_path(
                str(path),
                dpi=settings.ocr_dpi,
                first_page=first,
                last_page=last,
                fmt="png",
            )
        )

    def prepare(page: int, img: Any) -> None:
        images[page] = _prepare_image(img, max_dim, use_jpeg, jpeg_quality)
        logger.debug(f"Prepared page {page}: {len(images[page]) / 1024:.1f}KB")

    for first, last in _page_runs(sorted(pages), batch_size):
        pil_images = render(first, last)
        if len(pil_images) == last - first + 1:
            for page, img in zip(range(first, last + 1), pil_images, strict=True):
                prepare(page, img)
        else:
            # Poppler skipped a page, so the images no longer line up with
            # page numbers. Render the run one page at a time instead.
            logger.warning(
                f"Got {len(pil_images)} images for pages {first}-{last} of "
                f"{path.name}; rendering them one at a time"
            )
            del pil_images
            for page in range(first, last + 1):
                pil_images = render(page, page)
                if pil_images:
                    prepare(page, pil_images[0])

        # Free batch memory before next batch
        del pil_images

    return images


# ---------------------------------------------------------------------------
# OCR extraction
# ---------------------------------------------------------------------------

async def _ocr_pages(images: dict[int, bytes]) -> dict[int, str | VisionOCRError]:
    """OCR page images concurrently (up to ocr_concurrency at once).

    Returns each page's text, or the error its vision request raised.
    """
    client = VisionOCRClient()
    semaphore = asyncio.Semaphore(settings.ocr_concurrency)
    image_format = "jpeg" if settings.ocr_image_format.lower() == "jpeg" else "png"

    async def ocr_page(page_num: int, image_bytes: bytes) -> tuple[int, str | VisionOCRError]:
        async with semaphore:
            try:
                text = await client.ocr_image(
                    image_bytes, page_num=page_num, image_format=image_format
                )
                return page_num, text
            except VisionOCRError as e:
                logger.error(f"OCR failed for page {page_num}: {e}")
                return page_num, e

    try:
        results = await asyncio.gather(*(ocr_page(n, img) for n, img in images.items()))
    finally:
        await client.close()
    return dict(results)


async def extract_scanned_pdf(path: Path) -> ExtractedContent:
    """
    Extract text from a scanned PDF using vision LLM OCR, in a single pass.

    Pages whose OCR fails are left as ``[OCR failed for page N]`` placeholders
    and listed in ``metadata['ocr_failed_pages']``.

    Args:
        path: Path to scanned PDF

    Returns:
        ExtractedContent with OCR'd text

    Raises:
        ExtractionError: If the PDF cannot be rendered
    """
    logger.info(f"Starting vision LLM OCR for: {path.name}")
    page_count = _count_pages(path)
    progress = _OCRProgress(page_count)
    outcomes = await _ocr_pages(_render_pages(path, range(1, page_count + 1)))
    for page, outcome in outcomes.items():
        progress.record(page, outcome)
    return progress.to_content(exhausted=False)


def _all_pages_failed(result: ExtractedContent) -> bool:
    """Whether every page of an OCR result failed.

    Each page whose vision call errored is a ``[OCR failed for page N]``
    placeholder, with its number recorded in ``metadata['ocr_failed_pages']``.
    When every page failed the result text is nothing but placeholders, so it
    must not be accepted as document content or written to the cache.
    """
    page_count = result.page_count or 0
    failed = len(result.metadata.get("ocr_failed_pages", []))
    return page_count > 0 and failed >= page_count


def _is_final(cached: ExtractedContent) -> bool:
    """Whether a cached result may be served without running OCR again."""
    if not cached.metadata.get("ocr_failed_pages"):
        return True
    # Failed pages are final only once their attempts ran out. Older versions
    # cached partial results without that marker; those are resumed instead.
    return bool(cached.metadata.get("ocr_attempts_exhausted")) and not _all_pages_failed(cached)


async def _extract_with_cache(path: Path) -> ExtractedContent:
    """
    Extract a scanned PDF, keeping completed pages across failed attempts.

    A finished result is cached by file metadata. While pages keep failing,
    the completed ones are saved as progress and IncompleteOCRError is raised,
    so the next attempt only sends the remaining pages to the vision endpoint.
    A page is given up on, keeping its placeholder, after ocr_max_page_attempts
    failures charged to it; endpoint outages are not charged.

    Blocks while another extraction of the same file, in this process or
    another, is running; callers run this on its own loop in a worker thread.

    Raises:
        IncompleteOCRError: If pages failed and can still be retried
        ExtractionError: If the PDF cannot be rendered
    """
    if not settings.ocr_cache_enabled:
        # Without a cache there is nowhere to keep progress or attempt counts,
        # so a single pass decides the result, failed pages included.
        return await extract_scanned_pdf(path)

    cache_key = _get_cache_key(path)
    # Concurrent extractions of one file would start from the same progress
    # and overwrite each other's results, so they take turns. The lock is
    # released if its holder dies, and the holder's OCR is bounded by the
    # per-page timeout, so the wait needs no limit of its own.
    with file_lock(_get_cache_path(cache_key), timeout=math.inf):
        return await _resume_ocr(path, cache_key)


async def _resume_ocr(path: Path, cache_key: str) -> ExtractedContent:
    """Body of _extract_with_cache, run while holding the file's cache lock."""
    cached = _load_from_cache(cache_key)
    if cached is not None and _is_final(cached):
        return cached

    logger.info(f"Starting vision LLM OCR for: {path.name}")
    page_count = _count_pages(path)
    progress = (
        _load_progress(cache_key, page_count)
        or _progress_from_legacy(cached, page_count)
        or _OCRProgress(page_count)
    )

    max_attempts = max(1, settings.ocr_max_page_attempts)
    pending = progress.pending(max_attempts)
    if pending:
        images = _render_pages(path, pending)
        outcomes = await _ocr_pages(images)
        for page in pending:
            if page not in images:
                # Charge the page so a render that keeps coming back short
                # cannot keep the document retrying forever.
                logger.warning(f"Page {page} of {path.name} was not rendered")
                outcomes[page] = VisionOCRError(f"Page {page} was not rendered")
        for page, outcome in outcomes.items():
            progress.record(page, outcome)

    remaining = progress.pending(max_attempts)
    if remaining:
        try:
            _save_progress(cache_key, progress)
        except OSError as e:
            logger.error(f"Failed to save OCR progress {cache_key[:12]}...: {e}")
            raise IncompleteOCRError(path.name, remaining, page_count, e) from e
        raise IncompleteOCRError(path.name, remaining, page_count)

    result = progress.to_content(exhausted=True)
    # Do not cache a result in which every page failed: it is never accepted
    # as document content, and serving it from the cache would stop OCR from
    # being retried until the file's mtime changes.
    if not _all_pages_failed(result):
        _save_to_cache(cache_key, result)
    _remove_progress(cache_key)
    return result


def extract_scanned_pdf_sync(path: Path) -> ExtractedContent:
    """
    Synchronous entry point for scanned PDF extraction.

    Blocks the calling thread until OCR finishes. Async callers must run
    extraction on a worker thread (as the indexer and processing queue do)
    rather than on their event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop in this thread, the normal case on an extraction worker.
        return asyncio.run(_extract_with_cache(path))

    # asyncio.run() cannot nest inside a running loop, so OCR gets its own
    # thread. The caller's loop still waits on .result() until OCR finishes.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _extract_with_cache(path)).result()


def _merge_pages(page_texts: list[str]) -> str:
    """Merge OCR results from multiple pages."""
    if not page_texts:
        return ""

    if len(page_texts) == 1:
        return (page_texts[0] or "").strip()

    parts = []
    for i, text in enumerate(page_texts, 1):
        stripped = (text or "").strip()
        if stripped:
            parts.append(f"--- Page {i} ---\n\n{stripped}")

    return "\n\n".join(parts)
