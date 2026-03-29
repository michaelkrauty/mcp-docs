"""Vision LLM client for OCR via OpenAI-compatible endpoint."""

import base64
import logging
import time
from typing import Any, ClassVar

import httpx

from mcp_docs.settings import settings

logger = logging.getLogger(__name__)


class VisionOCRError(Exception):
    """Raised when vision OCR fails."""

    pass


class VisionOCRClient:
    """
    OCR via vision LLM using OpenAI-compatible API.

    Circuit breaker state is shared across all instances so that repeated
    failures from one document don't cause cascading timeouts in others.
    """

    # Class-level circuit breaker (shared across all instances)
    _failure_count: ClassVar[int] = 0
    _circuit_open_until: ClassVar[float | None] = None
    _max_failures: ClassVar[int] = 3
    _circuit_reset_seconds: ClassVar[int] = 60

    OCR_PROMPT = """Perform OCR on this document image. Extract ALL text exactly as it appears.

Output requirements:
1. Preserve the exact text content - do not paraphrase or summarize
2. Maintain document structure using markdown:
   - Use headers (#, ##, ###) for section titles
   - Use bullet points for lists
   - Use | for tables (markdown table format)
   - Use > for quoted text
   - Preserve paragraph breaks with blank lines
3. For forms, preserve field labels and their values
4. If text is unclear, use [unclear] marker
5. Do not add any commentary or explanation - just the extracted text

Begin OCR transcription:"""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (base_url or settings.ocr_vision_url).rstrip("/")
        self.model = model if model is not None else settings.ocr_vision_model
        self.timeout = float(timeout or settings.ocr_timeout)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=5.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @classmethod
    def _check_circuit_breaker(cls) -> None:
        """Check if circuit breaker is open and raise if so."""
        if cls._circuit_open_until is not None:
            if time.time() < cls._circuit_open_until:
                remaining = cls._circuit_open_until - time.time()
                raise VisionOCRError(
                    f"Vision OCR circuit breaker open. "
                    f"Service unavailable for {remaining:.0f}s after "
                    f"{cls._max_failures} consecutive failures."
                )
            cls._circuit_open_until = None
            cls._failure_count = 0

    @classmethod
    def _record_failure(cls) -> None:
        """Record a failure and potentially open circuit breaker."""
        cls._failure_count += 1
        if cls._failure_count >= cls._max_failures:
            cls._circuit_open_until = time.time() + cls._circuit_reset_seconds
            logger.warning(
                f"Vision OCR circuit breaker opened after {cls._failure_count} failures. "
                f"Will retry in {cls._circuit_reset_seconds}s"
            )

    @classmethod
    def _record_success(cls) -> None:
        """Record a success and reset failure count."""
        cls._failure_count = 0
        cls._circuit_open_until = None

    async def ocr_image(
        self,
        image_bytes: bytes,
        page_num: int = 1,
        image_format: str = "png",
    ) -> str:
        """
        Perform OCR on a single image using vision LLM.

        Args:
            image_bytes: PNG or JPEG image bytes
            page_num: Page number for context (used in logging)
            image_format: Image format (png or jpeg)

        Returns:
            Extracted text as markdown

        Raises:
            VisionOCRError: If OCR fails
        """
        self._check_circuit_breaker()

        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = "image/png" if image_format == "png" else "image/jpeg"
        image_url = f"data:{mime_type};base64,{base64_data}"

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": self.OCR_PROMPT},
                ],
            }
        ]

        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": 16384,
            "temperature": 0.0,
        }
        if self.model:
            payload["model"] = self.model

        client = await self._get_client()

        try:
            logger.debug(f"Sending OCR request for page {page_num} to {self.base_url}")
            start_time = time.time()

            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()

            data = resp.json()
            elapsed = time.time() - start_time

            content: str = data["choices"][0]["message"]["content"]
            logger.debug(
                f"OCR page {page_num} complete in {elapsed:.1f}s, "
                f"{len(content)} chars extracted"
            )

            self._record_success()
            return content

        except httpx.TimeoutException as e:
            self._record_failure()
            raise VisionOCRError(
                f"Vision OCR timeout for page {page_num} after {self.timeout}s. "
                f"Consider increasing DOCS_OCR_TIMEOUT."
            ) from e

        except httpx.HTTPStatusError as e:
            self._record_failure()
            error_detail = e.response.text[:500] if e.response.text else "No details"
            raise VisionOCRError(
                f"Vision OCR failed for page {page_num}: "
                f"HTTP {e.response.status_code} - {error_detail}"
            ) from e

        except httpx.ConnectError as e:
            self._record_failure()
            raise VisionOCRError(
                f"Cannot connect to vision OCR server at {self.base_url}. "
                f"Is the server running? Error: {e}"
            ) from e

        except Exception as e:
            self._record_failure()
            raise VisionOCRError(f"Vision OCR error for page {page_num}: {e}") from e

    async def health_check(self) -> bool:
        """Check if the vision server is reachable."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
