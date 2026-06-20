"""OCR extraction for scanned PDFs using vision LLMs."""

import asyncio
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

from vector_core.settings import settings as vector_settings

from mcp_docs.extraction.vision_client import VisionOCRClient, VisionOCRError
from mcp_docs.models import ExtractedContent, ExtractionError
from mcp_docs.settings import settings

logger = logging.getLogger(__name__)


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
        with open(cache_path, "w") as f:
            json.dump(data, f)
        logger.debug(f"Saved OCR result to cache: {cache_key[:12]}...")
    except Exception as e:
        logger.warning(f"Failed to save OCR cache {cache_key}: {e}")


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


def pdf_to_images(path: Path, batch_size: int = 20) -> list[bytes]:
    """
    Convert PDF pages to compressed image bytes in batches.

    Processes pages in batches to limit peak memory — only one batch of
    raw PIL images is in memory at a time. The returned list contains
    compressed JPEG/PNG bytes which are much smaller.

    Args:
        path: Path to PDF file
        batch_size: Pages to render per batch (controls peak memory)

    Returns:
        List of image bytes, one per page

    Raises:
        ExtractionError: If conversion fails or pdf2image not installed
    """
    try:
        from pdf2image import convert_from_path, pdfinfo_from_path
    except ImportError as e:
        raise ExtractionError(
            "pdf2image not installed. Install with: pip install mcp-docs[ocr]\n"
            "Also requires poppler: apt install poppler-utils (Linux) "
            "or brew install poppler (macOS)"
        ) from e

    try:
        info = pdfinfo_from_path(str(path))
        total_pages = info["Pages"]
    except Exception as e:
        raise ExtractionError(f"Failed to read PDF info: {e}") from e

    if total_pages == 0:
        raise ExtractionError(f"No pages found in PDF: {path.name}")

    pages_to_process = min(total_pages, settings.ocr_max_pages)
    if total_pages > settings.ocr_max_pages:
        logger.warning(
            f"PDF has {total_pages} pages, limiting OCR to first {settings.ocr_max_pages}"
        )

    use_jpeg = settings.ocr_image_format.lower() == "jpeg"
    max_dim = settings.ocr_image_max_dimension
    jpeg_quality = settings.ocr_jpeg_quality
    image_bytes_list: list[bytes] = []

    for batch_start in range(1, pages_to_process + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, pages_to_process)

        pil_images = convert_from_path(
            str(path),
            dpi=settings.ocr_dpi,
            first_page=batch_start,
            last_page=batch_end,
            fmt="png",
        )

        for i, img in enumerate(pil_images):
            page_num = batch_start + i
            img_bytes = _prepare_image(img, max_dim, use_jpeg, jpeg_quality)
            image_bytes_list.append(img_bytes)
            logger.debug(
                f"Prepared page {page_num}/{pages_to_process}: "
                f"{len(img_bytes) / 1024:.1f}KB"
            )

        # Free batch memory before next batch
        del pil_images

    return image_bytes_list


# ---------------------------------------------------------------------------
# OCR extraction
# ---------------------------------------------------------------------------

async def extract_scanned_pdf(path: Path) -> ExtractedContent:
    """
    Extract text from a scanned PDF using vision LLM OCR.

    Processes pages concurrently (up to ocr_concurrency at once).

    Args:
        path: Path to scanned PDF

    Returns:
        ExtractedContent with OCR'd text

    Raises:
        ExtractionError: If OCR fails
    """
    logger.info(f"Starting vision LLM OCR for: {path.name}")

    images = pdf_to_images(path)
    page_count = len(images)
    logger.info(f"Converted {page_count} pages to images at {settings.ocr_dpi} DPI")

    client = VisionOCRClient()
    semaphore = asyncio.Semaphore(settings.ocr_concurrency)
    image_format = "jpeg" if settings.ocr_image_format.lower() == "jpeg" else "png"

    async def ocr_page(page_num: int, image_bytes: bytes) -> tuple[int, str, bool]:
        """Returns (page_num, text, failed)."""
        async with semaphore:
            try:
                text = await client.ocr_image(
                    image_bytes, page_num=page_num, image_format=image_format
                )
                return (page_num, text, False)
            except VisionOCRError as e:
                logger.error(f"OCR failed for page {page_num}: {e}")
                return (page_num, f"[OCR failed for page {page_num}]", True)

    try:
        tasks = [ocr_page(i, img) for i, img in enumerate(images, 1)]
        results = await asyncio.gather(*tasks)
    finally:
        await client.close()

    # Sort by page number and separate results
    results_sorted = sorted(results, key=lambda x: x[0])
    page_texts = [text for _, text, _ in results_sorted]
    failed_pages = [num for num, _, failed in results_sorted if failed]

    # Combine pages
    full_text = _merge_pages(page_texts)
    word_count = len(full_text.split()) if full_text else 0

    metadata = {
        "ocr_method": "vision_llm",
        "ocr_dpi": settings.ocr_dpi,
    }
    if settings.ocr_vision_model:
        metadata["ocr_model"] = settings.ocr_vision_model
    if failed_pages:
        metadata["ocr_failed_pages"] = failed_pages

    logger.info(
        f"OCR complete: {word_count} words from {page_count} pages "
        f"({len(failed_pages)} failed)"
    )

    return ExtractedContent(
        text=full_text,
        title=None,
        page_count=page_count,
        word_count=word_count,
        metadata=metadata,
    )


def _all_pages_failed(result: ExtractedContent) -> bool:
    """Whether every page of an OCR result failed.

    ``ocr_page`` emits a ``[OCR failed for page N]`` placeholder for each page
    whose vision call errored, recording the page number in
    ``metadata['ocr_failed_pages']``. When every page failed the result text is
    nothing but placeholders, so it must not be accepted as document content or
    written to the cache.
    """
    page_count = result.page_count or 0
    failed = len(result.metadata.get("ocr_failed_pages", []))
    return page_count > 0 and failed >= page_count


async def _extract_with_cache(path: Path) -> ExtractedContent:
    """Extract scanned PDF with caching."""
    if not settings.ocr_cache_enabled:
        return await extract_scanned_pdf(path)

    cache_key = _get_cache_key(path)
    cached = _load_from_cache(cache_key)
    # Ignore a cached fully-failed result (for example one written before this
    # guard existed) so OCR is retried instead of serving placeholder text.
    if cached is not None and not _all_pages_failed(cached):
        return cached

    result = await extract_scanned_pdf(path)
    # Do not cache a fully-failed OCR result: its body is only
    # "[OCR failed for page N]" placeholders, and the cache key is the file's
    # path/size/mtime, so caching it would poison every future extraction of
    # this file until its mtime changes.
    if not _all_pages_failed(result):
        _save_to_cache(cache_key, result)
    return result


def extract_scanned_pdf_sync(path: Path) -> ExtractedContent:
    """
    Synchronous entry point for scanned PDF extraction.

    Handles the async/sync boundary cleanly — works whether called
    from a sync context or from within an async event loop.
    """
    try:
        asyncio.get_running_loop()
        # Already in an async context — run OCR in a separate thread
        # with its own event loop to avoid blocking the caller's loop.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _extract_with_cache(path)).result()
    except RuntimeError:
        # No running event loop — safe to create one
        return asyncio.run(_extract_with_cache(path))


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
