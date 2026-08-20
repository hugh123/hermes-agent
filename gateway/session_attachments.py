"""Session-scoped file staging for the HTTP API server.

The API server and the TUI run the same agent core, including remote terminal
backends whose filesystem differs from the gateway host.  Files staged here
live below the profile-aware ``HERMES_HOME/attachments`` cache directory, which
the existing credential-file machinery mounts or syncs into those backends.

This module intentionally owns storage mechanics only.  Chat endpoints keep
their existing string/multimodal contract; callers may place the returned
agent-visible path in the message they send to Hermes.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from hermes_constants import get_hermes_home


MAX_SESSION_ATTACHMENT_BYTES = 50 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


class AttachmentUploadError(ValueError):
    """Base class for caller-correctable attachment upload failures."""


class AttachmentTooLargeError(AttachmentUploadError):
    """Raised when the decoded upload exceeds the configured byte limit."""


class EmptyAttachmentError(AttachmentUploadError):
    """Raised when an upload contains no file bytes."""


@dataclass(frozen=True)
class SessionAttachmentTarget:
    """Allocated storage target for one API-server session attachment."""

    attachment_id: str
    session_id: str
    filename: str
    content_type: str
    host_path: Path
    agent_path: str


@dataclass(frozen=True)
class StoredSessionAttachment:
    """Metadata produced after an attachment is atomically committed."""

    target: SessionAttachmentTarget
    size: int
    sha256: str


def sanitize_attachment_name(name: str) -> str:
    """Return a filename-only, control-character-free attachment name."""

    # User agents and multipart libraries may submit client paths using
    # POSIX separators, Windows separators, or percent-encoded separators.
    # Decode once, then discard every untrusted parent component.
    decoded = unquote(str(name or "").strip())
    candidate = Path(decoded.replace("\\", "/")).name
    candidate = re.sub(r"[\x00-\x1f\x7f]+", "_", candidate)
    candidate = candidate.strip().strip(".")
    return candidate or "attachment"


def unique_attachment_path(root: Path, filename: str) -> Path:
    """Return a non-existing sibling path without overwriting prior files."""

    candidate = root / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem or "attachment"
    suffix = Path(filename).suffix
    counter = 2
    while True:
        candidate = root / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _session_storage_key(session_id: str) -> str:
    """Create a filesystem-safe, non-reversible namespace for a session ID."""

    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]


def _normalized_content_type(filename: str, content_type: str | None) -> str:
    """Normalize untrusted multipart MIME metadata with a filename fallback."""

    supplied = str(content_type or "").split(";", 1)[0].strip().lower()
    if supplied and re.fullmatch(r"[a-z0-9!#$&^_.+\-]+/[a-z0-9!#$&^_.+\-]+", supplied):
        return supplied[:255]
    guessed, _encoding = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def agent_visible_attachment_path(host_path: Path) -> str:
    """Translate a staged host path to the active terminal backend's path."""

    try:
        from tools.terminal_tool import _ensure_terminal_env_bridged

        _ensure_terminal_env_bridged()
        from tools.credential_files import to_agent_visible_cache_path

        return to_agent_visible_cache_path(str(host_path.resolve()))
    except Exception:
        return str(host_path.resolve())


def allocate_session_attachment(
    session_id: str,
    filename: str,
    content_type: str | None = None,
) -> SessionAttachmentTarget:
    """Allocate an isolated target below the active profile's attachment root."""

    clean_name = sanitize_attachment_name(filename)
    attachment_id = f"att_{uuid.uuid4().hex}"
    root = (
        get_hermes_home()
        / "attachments"
        / "api_server"
        / _session_storage_key(session_id)
        / attachment_id
    )
    root.mkdir(parents=True, exist_ok=False)
    try:
        host_path = unique_attachment_path(root, clean_name).resolve()
        return SessionAttachmentTarget(
            attachment_id=attachment_id,
            session_id=session_id,
            filename=host_path.name,
            content_type=_normalized_content_type(host_path.name, content_type),
            host_path=host_path,
            agent_path=agent_visible_attachment_path(host_path),
        )
    except BaseException:
        with suppress(OSError):
            root.rmdir()
        raise


class SessionAttachmentWriter:
    """Stream an attachment to a sibling temp file and atomically commit it."""

    def __init__(
        self,
        target: SessionAttachmentTarget,
        *,
        max_bytes: int | None = None,
    ) -> None:
        self.target = target
        self.max_bytes = MAX_SESSION_ATTACHMENT_BYTES if max_bytes is None else max_bytes
        self.size = 0
        self._sha256 = hashlib.sha256()
        self._committed = False
        self._closed = False
        tmp_fd = -1
        tmp_path: Path | None = None
        try:
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target.filename}.",
                suffix=".upload",
                dir=str(target.host_path.parent),
            )
            tmp_path = Path(tmp_name)
            self._stream = os.fdopen(tmp_fd, "wb")
            tmp_fd = -1  # Ownership transferred to the buffered file object.
            self._tmp_path = tmp_path
        except BaseException:
            if tmp_fd >= 0:
                os.close(tmp_fd)
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            with suppress(OSError):
                target.host_path.parent.rmdir()
            raise

    def __enter__(self) -> "SessionAttachmentWriter":
        return self

    def write(self, chunk: bytes) -> None:
        """Append one decoded multipart chunk while enforcing the byte cap."""

        if self._closed:
            raise RuntimeError("attachment writer is closed")
        if not chunk:
            return
        next_size = self.size + len(chunk)
        if next_size > self.max_bytes:
            raise AttachmentTooLargeError(
                f"attachment exceeds the {self.max_bytes}-byte upload limit"
            )
        self._stream.write(chunk)
        self._sha256.update(chunk)
        self.size = next_size

    def commit(self) -> StoredSessionAttachment:
        """Flush and atomically publish the upload, rejecting empty files."""

        if self._closed:
            raise RuntimeError("attachment writer is closed")
        if self.size == 0:
            raise EmptyAttachmentError("attachment file is empty")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._closed = True
        os.replace(self._tmp_path, self.target.host_path)
        self._committed = True
        return StoredSessionAttachment(
            target=self.target,
            size=self.size,
            sha256=self._sha256.hexdigest(),
        )

    def close(self) -> None:
        """Close and remove every partial artifact from an unsuccessful upload."""

        if not self._closed:
            # Cleanup must not mask the upload error that caused the context
            # manager to abort (for example, size-limit or cancellation).
            with suppress(OSError):
                self._stream.close()
            self._closed = True
        if not self._committed:
            with suppress(OSError):
                self._tmp_path.unlink(missing_ok=True)
            with suppress(OSError):
                self.target.host_path.unlink(missing_ok=True)
            with suppress(OSError):
                self.target.host_path.parent.rmdir()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()
