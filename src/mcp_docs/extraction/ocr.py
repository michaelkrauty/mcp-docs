"""OCR extraction for scanned PDFs using vision LLMs."""

import hashlib
import io
import json
import logging
from pathlib import Path

from PIL import Image

from mcp_docs.extraction.vision_client import VisionOCRClient, VisionOCRError
from mcp_docs.models import ExtractedContent, ExtractionError
from mcp_docs.settings import settings
from vector_core.settings import settings as vector_settings

logger = logging.getLogger(__name__)


def _get_cache_dir() -> Path:
    """Get the OCR cache directory, creating if needed."""
    cache_dir = vector_settings.shared_data_dir / "ocr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_path(content_hash: str) -> Path:
    """Get the cache file path for a content hash."""
    return _get_cache_dir() / f"{content_hash}.json"


def _load_from_cache(content_hash: str) -> ExtractedContent | None:
    """
    Load OCR result from cache if available.

    Args:
        content_hash: SHA-256 hash of the PDF content

    Returns:
        ExtractedContent if cached, None otherwise
    """
    cache_path = _get_cache_path(content_hash)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path) as f:
            data = json.load(f)

        logger.info(f"Loaded OCR result from cache: {content_hash[:12]}...")
        return ExtractedContent(
            text=data["text"],
            title=data.get("title"),
            page_count=data.get("page_count", 0),
            word_count=data.get("word_count", 0),
            metadata=data.get("metadata", {}),
        )
    except Exception as e:
        logger.warning(f"Failed to load OCR cache {content_hash}: {e}")
        return None


def _save_to_cache(content_hash: str, result: ExtractedContent) -> None:
    """
    Save OCR result to cache.

    Args:
        content_hash: SHA-256 hash of the PDF content
        result: Extraction result to cache
    """
    cache_path = _get_cache_path(content_hash)
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
        logger.debug(f"Saved OCR result to cache: {content_hash[:12]}...")
    except Exception as e:
        logger.warning(f"Failed to save OCR cache {content_hash}: {e}")


def _compute_content_hash(path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def pdf_to_images(path: Path) -> list[bytes]:
    """
    Convert PDF pages to PNG images.

    Args:
        path: Path to PDF file

    Returns:
        List of PNG image bytes, one per page

    Raises:
        ExtractionError: If conversion fails
    """
    try:
        from pdf2image import convert_from_path
    except ImportError as e:
        raise ExtractionError(
            "pdf2image not installed. Install with: uv pip install pdf2image\n"
            "Also requires poppler: apt install poppler-utils (Linux) "
            "or brew install poppler (Mac)"
        ) from e

    try:
        # Convert PDF to PIL images at high DPI
        pil_images = convert_from_path(
            path,
            dpi=settings.ocr_dpi,
            fmt="png",
        )
    except Exception as e:
        raise ExtractionError(f"Failed to convert PDF to images: {e}") from e

    if not pil_images:
        raise ExtractionError(f"No pages found in PDF: {path.name}")

    # Limit pages for safety
    total_pages = len(pil_images)
    if total_pages > settings.ocr_max_pages:
        logger.warning(
            f"PDF has {total_pages} pages, limiting OCR to first {settings.ocr_max_pages}"
        )
        pil_images = pil_images[: settings.ocr_max_pages]

    # Resize if needed and convert to bytes
    image_bytes_list: list[bytes] = []
    use_jpeg = settings.ocr_image_format.lower() == "jpeg"

    for i, img in enumerate(pil_images, 1):
        # Resize if exceeds max dimension
        img = _resize_if_needed(img, settings.ocr_image_max_dimension)

        # Convert to bytes (JPEG for smaller size, PNG for lossless)
        buffer = io.BytesIO()
        if use_jpeg:
            # Convert to RGB if needed (JPEG doesn't support alpha)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            img.save(buffer, format="JPEG", quality=settings.ocr_jpeg_quality)
        else:
            img.save(buffer, format="PNG", optimize=True)

        image_bytes = buffer.getvalue()
        image_bytes_list.append(image_bytes)

        logger.debug(
            f"Prepared page {i}/{len(pil_images)}: "
            f"{img.size[0]}x{img.size[1]}, {len(image_bytes) / 1024:.1f}KB "
            f"({'JPEG' if use_jpeg else 'PNG'})"
        )

    return image_bytes_list


def _resize_if_needed(img: Image.Image, max_dim: int) -> Image.Image:
    """Resize image if any dimension exceeds max_dim, preserving aspect ratio."""
    width, height = img.size
    if width <= max_dim and height <= max_dim:
        return img

    # Calculate new size preserving aspect ratio
    if width > height:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    else:
        new_height = max_dim
        new_width = int(width * (max_dim / height))

    logger.debug(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


async def extract_scanned_pdf(path: Path) -> ExtractedContent:
    """
    Extract text from a scanned PDF using vision LLM OCR.

    Args:
        path: Path to scanned PDF

    Returns:
        ExtractedContent with OCR'd text

    Raises:
        ExtractionError: If OCR fails
    """
    logger.info(f"Starting vision LLM OCR for: {path.name}")

    # Convert PDF to images
    images = pdf_to_images(path)
    page_count = len(images)
    logger.info(f"Converted {page_count} pages to images at {settings.ocr_dpi} DPI")

    # OCR each page
    client = VisionOCRClient()
    try:
        page_texts: list[str] = []
        failed_pages: list[int] = []

        for i, image_bytes in enumerate(images, 1):
            logger.info(f"OCR processing page {i}/{page_count}...")
            try:
                image_format = "jpeg" if settings.ocr_image_format.lower() == "jpeg" else "png"
                text = await client.ocr_image(image_bytes, page_num=i, image_format=image_format)
                page_texts.append(text)
            except VisionOCRError as e:
                logger.error(f"OCR failed for page {i}: {e}")
                page_texts.append(f"[OCR failed for page {i}]")
                failed_pages.append(i)

    finally:
        await client.close()

    # Combine pages
    full_text = _merge_pages(page_texts)
    word_count = len(full_text.split()) if full_text else 0

    # Build metadata
    metadata = {
        "ocr_method": "vision_llm",
        "ocr_model": settings.ocr_vision_model,
        "ocr_dpi": settings.ocr_dpi,
    }
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


async def extract_scanned_pdf_cached(path: Path) -> ExtractedContent:
    """
    Extract text from scanned PDF with caching.

    Checks cache first, performs OCR if not cached, saves result to cache.

    Args:
        path: Path to scanned PDF

    Returns:
        ExtractedContent with OCR'd text
    """
    if not settings.ocr_cache_enabled:
        return await extract_scanned_pdf(path)

    # Check cache
    content_hash = _compute_content_hash(path)
    cached = _load_from_cache(content_hash)
    if cached is not None:
        return cached

    # Perform OCR
    result = await extract_scanned_pdf(path)

    # Save to cache
    _save_to_cache(content_hash, result)

    return result


def _merge_pages(page_texts: list[str]) -> str:
    """Merge OCR results from multiple pages."""
    if not page_texts:
        return ""

    if len(page_texts) == 1:
        return page_texts[0].strip()

    # Add page markers for multi-page documents
    parts = []
    for i, text in enumerate(page_texts, 1):
        text = text.strip()
        if text:
            parts.append(f"--- Page {i} ---\n\n{text}")

    return "\n\n".join(parts)
