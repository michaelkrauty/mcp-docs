"""Filesystem management tools for mcp-docs.

Tools:
- move_file: Move a file and update document registry
- create_directory: Create a directory
- rename_directory: Rename a directory and update all document paths under it
- move_directory: Move a directory and update all document paths under it
- delete_directory: Delete an empty directory (no files allowed)
"""

import logging
import shutil
from pathlib import Path

from vector_core.errors import ErrorCode, error_response

from mcp_docs.app import mcp
from mcp_docs.singletons import (
    get_document_indexer,
    get_document_processor,
    get_document_store,
)

logger = logging.getLogger(__name__)


def _validate_within_root(path: Path, store) -> tuple[bool, str | None]:
    """
    Validate that path is within a document root.

    Args:
        path: Path to validate
        store: DocumentStore instance

    Returns:
        (is_valid, root_path_or_error)
    """
    if not path.is_absolute():
        return False, "Path must be absolute"

    # Check if path is within any document root
    roots = store.list_roots()
    for root in roots:
        root_path = Path(root.path)
        try:
            path.relative_to(root_path)
            return True, root.path
        except ValueError:
            continue

    return False, "Path is not within any document root"


def _find_document_root(path: Path, store) -> str | None:
    """Find which document root contains this path."""
    roots = store.list_roots()
    for root in roots:
        root_path = Path(root.path)
        try:
            path.relative_to(root_path)
            return root.path
        except ValueError:
            continue
    return None


@mcp.tool()
async def move_file(source_path: str, destination_path: str) -> dict:
    """
    Move a file and update document registry.

    Args:
        source_path: Absolute path to source file
        destination_path: Absolute path to destination file

    Returns:
        Success dict with old_path, new_path, document_id
    """
    try:
        store = get_document_store()
        processor = await get_document_processor()
        indexer = await get_document_indexer()

        # 1. Validate paths absolute, source exists and is file
        source = Path(source_path).resolve()
        dest = Path(destination_path).resolve()

        if not source.is_absolute() or not dest.is_absolute():
            return error_response(ErrorCode.INVALID_INPUT, "Paths must be absolute")

        if not source.exists():
            return error_response(ErrorCode.FILE_NOT_FOUND, f"Source file does not exist: {source}")

        if not source.is_file():
            return error_response(ErrorCode.INVALID_INPUT, f"Source is not a file: {source}")

        # 2. Get document by path, error if not registered
        document = store.get_by_path(str(source))
        if document is None:
            return error_response(ErrorCode.NOT_FOUND, f"Document not registered: {source}")

        # 3. Validate destination parent exists, dest doesn't exist
        dest_parent = dest.parent
        if not dest_parent.exists():
            return error_response(
                ErrorCode.INVALID_INPUT,
                f"Destination directory does not exist: {dest_parent}",
            )

        if dest.exists():
            return error_response(ErrorCode.CONFLICT, f"Destination already exists: {dest}")

        # 4. Validate both paths within document roots
        source_valid, source_root = _validate_within_root(source, store)
        if not source_valid:
            return error_response(
                ErrorCode.PERMISSION_DENIED,
                f"Source path not in document root: {source_root}",
            )

        dest_valid, dest_root_result = _validate_within_root(dest, store)
        if not dest_valid:
            return error_response(
                ErrorCode.PERMISSION_DENIED,
                f"Destination path not in document root: {dest_root_result}",
            )

        dest_root = _find_document_root(dest, store)

        # 5. Wait for document processing to complete
        processing_complete = await processor.wait_for_documents([document.id], timeout=60.0)
        if not processing_complete:
            return error_response(
                ErrorCode.TIMEOUT,
                "Document processing did not complete within timeout",
            )

        # 6. Move the file
        try:
            shutil.move(str(source), str(dest))
        except OSError as e:
            return error_response(ErrorCode.INVALID_INPUT, f"Failed to move file: {e}")

        # 7. Update document path and document_root
        store.update(document.id, path=str(dest), document_root=dest_root)

        # 8. Update vector index path
        await indexer.update_document_path_in_index(document.id, str(dest))

        logger.info(f"Moved file {source} -> {dest}")

        return {
            "success": True,
            "old_path": str(source),
            "new_path": str(dest),
            "document_id": str(document.id),
            "new_document_root": dest_root,
        }

    except Exception as e:
        logger.error(f"Error moving file: {e}")
        return error_response(ErrorCode.INTERNAL_ERROR, f"Internal error: {e}")


@mcp.tool()
async def create_directory(path: str, parents: bool = False) -> dict:
    """
    Create a directory.

    Args:
        path: Absolute path to directory to create
        parents: Whether to create parent directories if they don't exist

    Returns:
        Success dict with path, created flag
    """
    try:
        store = get_document_store()

        # 1. Validate path absolute, within document root
        dir_path = Path(path).resolve()

        if not dir_path.is_absolute():
            return error_response(ErrorCode.INVALID_INPUT, "Path must be absolute")

        valid, root_or_error = _validate_within_root(dir_path, store)
        if not valid:
            return error_response(
                ErrorCode.PERMISSION_DENIED,
                f"Path not in document root: {root_or_error}",
            )

        # 2. If exists, return success (idempotent)
        if dir_path.exists():
            if dir_path.is_dir():
                return {
                    "success": True,
                    "path": str(dir_path),
                    "created": False,
                    "message": "Directory already exists",
                }
            else:
                return error_response(
                    ErrorCode.CONFLICT,
                    f"Path exists but is not a directory: {dir_path}",
                )

        # 3. Create directory
        try:
            dir_path.mkdir(parents=parents)
        except OSError as e:
            return error_response(ErrorCode.INVALID_INPUT, f"Failed to create directory: {e}")

        logger.info(f"Created directory: {dir_path}")

        return {
            "success": True,
            "path": str(dir_path),
            "created": True,
        }

    except Exception as e:
        logger.error(f"Error creating directory: {e}")
        return error_response(ErrorCode.INTERNAL_ERROR, f"Internal error: {e}")


@mcp.tool()
async def rename_directory(path: str, new_name: str) -> dict:
    """
    Rename a directory and update all document paths under it.

    Args:
        path: Absolute path to directory to rename
        new_name: New directory name (not a path, just the name)

    Returns:
        Success dict with old_path, new_path, documents_updated count
    """
    try:
        store = get_document_store()
        processor = await get_document_processor()
        indexer = await get_document_indexer()

        # 1. Validate path exists and is directory
        dir_path = Path(path).resolve()

        if not dir_path.exists():
            return error_response(ErrorCode.FILE_NOT_FOUND, f"Directory does not exist: {dir_path}")

        if not dir_path.is_dir():
            return error_response(ErrorCode.INVALID_INPUT, f"Path is not a directory: {dir_path}")

        # 2. Validate new_name has no path separators
        if "/" in new_name or "\\" in new_name:
            return error_response(
                ErrorCode.INVALID_INPUT,
                "New name must not contain path separators",
            )

        # 3. Query all documents under path prefix
        docs_to_update = store.query_by_path_prefix(str(dir_path) + "/")

        if docs_to_update:
            # 4. Wait for any processing docs to complete
            doc_ids = [doc.id for doc in docs_to_update]
            processing_complete = await processor.wait_for_documents(doc_ids, timeout=120.0)
            if not processing_complete:
                return error_response(
                    ErrorCode.TIMEOUT,
                    "Document processing did not complete within timeout",
                )

        # 5. Calculate new_path = parent / new_name
        new_path = dir_path.parent / new_name

        # 6. Validate new_path doesn't exist
        if new_path.exists():
            return error_response(ErrorCode.CONFLICT, f"Destination already exists: {new_path}")

        # 7. Move directory
        try:
            shutil.move(str(dir_path), str(new_path))
        except OSError as e:
            return error_response(ErrorCode.INVALID_INPUT, f"Failed to rename directory: {e}")

        # 8. Update document paths in batch
        old_prefix = str(dir_path)
        new_prefix = str(new_path)
        docs_updated = store.update_paths_batch(old_prefix, new_prefix)

        # 9. Update document_root if needed
        new_root = _find_document_root(new_path, store)
        if new_root:
            store.update_document_roots_batch(new_prefix, new_root)

        # 10. Update vector index paths in batch
        await indexer.update_paths_batch_in_index(old_prefix, new_prefix)

        logger.info(f"Renamed directory {dir_path} -> {new_path}, updated {docs_updated} documents")

        return {
            "success": True,
            "old_path": str(dir_path),
            "new_path": str(new_path),
            "documents_updated": docs_updated,
        }

    except Exception as e:
        logger.error(f"Error renaming directory: {e}")
        return error_response(ErrorCode.INTERNAL_ERROR, f"Internal error: {e}")


@mcp.tool()
async def move_directory(source_path: str, destination_path: str) -> dict:
    """
    Move a directory and update all document paths under it.

    Args:
        source_path: Absolute path to source directory
        destination_path: Absolute path to destination directory

    Returns:
        Success dict with old_path, new_path, documents_updated count
    """
    try:
        store = get_document_store()
        processor = await get_document_processor()
        indexer = await get_document_indexer()

        # 1. Validate paths
        source = Path(source_path).resolve()
        dest = Path(destination_path).resolve()

        if not source.exists():
            return error_response(
                ErrorCode.FILE_NOT_FOUND,
                f"Source directory does not exist: {source}",
            )

        if not source.is_dir():
            return error_response(ErrorCode.INVALID_INPUT, f"Source is not a directory: {source}")

        # 2. Validate destination parent exists and dest doesn't exist
        dest_parent = dest.parent
        if not dest_parent.exists():
            return error_response(
                ErrorCode.INVALID_INPUT,
                f"Destination parent does not exist: {dest_parent}",
            )

        if dest.exists():
            return error_response(ErrorCode.CONFLICT, f"Destination already exists: {dest}")

        # 3. Query all documents under source path
        docs_to_update = store.query_by_path_prefix(str(source) + "/")

        if docs_to_update:
            # 4. Wait for any processing docs to complete
            doc_ids = [doc.id for doc in docs_to_update]
            processing_complete = await processor.wait_for_documents(doc_ids, timeout=120.0)
            if not processing_complete:
                return error_response(
                    ErrorCode.TIMEOUT,
                    "Document processing did not complete within timeout",
                )

        # 5. Move directory
        try:
            shutil.move(str(source), str(dest))
        except OSError as e:
            return error_response(ErrorCode.INVALID_INPUT, f"Failed to move directory: {e}")

        # 6. Update document paths in batch
        old_prefix = str(source)
        new_prefix = str(dest)
        docs_updated = store.update_paths_batch(old_prefix, new_prefix)

        # 7. Update document_root if needed
        new_root = _find_document_root(dest, store)
        if new_root:
            store.update_document_roots_batch(new_prefix, new_root)

        # 8. Update vector index paths in batch
        await indexer.update_paths_batch_in_index(old_prefix, new_prefix)

        logger.info(f"Moved directory {source} -> {dest}, updated {docs_updated} documents")

        return {
            "success": True,
            "old_path": str(source),
            "new_path": str(dest),
            "documents_updated": docs_updated,
        }

    except Exception as e:
        logger.error(f"Error moving directory: {e}")
        return error_response(ErrorCode.INTERNAL_ERROR, f"Internal error: {e}")


@mcp.tool()
async def delete_directory(path: str, recursive: bool = False) -> dict:
    """
    Delete an empty directory (no files allowed).

    Args:
        path: Absolute path to directory to delete
        recursive: Whether to delete subdirectories (only if they're empty of files)

    Returns:
        Success dict with path
    """
    try:
        store = get_document_store()

        # 1. Validate path exists and is directory
        dir_path = Path(path).resolve()

        if not dir_path.exists():
            return error_response(ErrorCode.FILE_NOT_FOUND, f"Directory does not exist: {dir_path}")

        if not dir_path.is_dir():
            return error_response(ErrorCode.INVALID_INPUT, f"Path is not a directory: {dir_path}")

        # 2. Validate not a document root
        roots = store.list_roots()
        for root in roots:
            if Path(root.path).resolve() == dir_path:
                return error_response(
                    ErrorCode.CONFLICT,
                    f"Cannot delete document root: {dir_path}",
                )

        # 3. Query documents under path - error if any exist
        docs_in_dir = store.query_by_path_prefix(str(dir_path) + "/")
        if docs_in_dir:
            return error_response(
                ErrorCode.CONFLICT,
                f"Directory contains {len(docs_in_dir)} registered documents, cannot delete"
            )

        # 4. If recursive: walk and verify all subdirs empty of files
        if recursive:
            for item in dir_path.rglob("*"):
                if item.is_file():
                    return error_response(
                        ErrorCode.CONFLICT,
                        f"Directory tree contains files, cannot delete (found: {item})"
                    )
        else:
            # Check only immediate children for non-recursive
            for item in dir_path.iterdir():
                if item.is_file():
                    return error_response(
                        ErrorCode.CONFLICT,
                        f"Directory contains files, cannot delete (found: {item})"
                    )

        # 5. Delete directory
        try:
            if recursive:
                shutil.rmtree(str(dir_path))
            else:
                dir_path.rmdir()
        except OSError as e:
            return error_response(ErrorCode.INVALID_INPUT, f"Failed to delete directory: {e}")

        logger.info(f"Deleted directory: {dir_path}")

        return {
            "success": True,
            "path": str(dir_path),
            "recursive": recursive,
        }

    except Exception as e:
        logger.error(f"Error deleting directory: {e}")
        return error_response(ErrorCode.INTERNAL_ERROR, f"Internal error: {e}")
