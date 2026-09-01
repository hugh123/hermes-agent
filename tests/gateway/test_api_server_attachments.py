"""Behavior tests for session-scoped HTTP attachment uploads."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from gateway import session_attachments
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra=extra))
    adapter._get_existing_session_or_404 = AsyncMock(
        return_value=({"id": "session-1"}, None)
    )
    return adapter


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(
        "/api/sessions/{session_id}/attachments",
        adapter._handle_session_attachment_upload,
    )
    app.router.add_post(
        "/api/sessions/{session_id}/attachments/url",
        adapter._handle_session_attachment_url_upload,
    )
    return app


def _file_form(
    content: bytes,
    *,
    filename: str = "contract.pdf",
    field_name: str = "file",
    content_type: str = "application/pdf",
) -> FormData:
    form = FormData()
    form.add_field(
        field_name,
        content,
        filename=filename,
        content_type=content_type,
    )
    return form


@pytest.mark.asyncio
async def test_upload_streams_file_and_returns_agent_path(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    payload = b"%PDF-1.7\nreal-pdf-payload"
    adapter = _make_adapter()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/session-1/attachments",
            data=_file_form(payload),
        )
        assert response.status == 201
        data = await response.json()

    stored_path = Path(data["path"])
    assert stored_path.read_bytes() == payload
    assert stored_path.is_relative_to(hermes_home / "attachments" / "api_server")
    assert data == {
        "object": "hermes.session.attachment",
        "id": data["id"],
        "session_id": "session-1",
        "filename": "contract.pdf",
        "content_type": "application/pdf",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "path": str(stored_path),
    }
    assert data["id"].startswith("att_")
    assert "host_path" not in data


@pytest.mark.asyncio
async def test_upload_sanitizes_client_path_and_never_overwrites(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    adapter = _make_adapter()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        first = await client.post(
            "/api/sessions/session-1/attachments",
            data=_file_form(b"first", filename="../../contract.pdf"),
        )
        second = await client.post(
            "/api/sessions/session-1/attachments",
            data=_file_form(b"second", filename=r"C:\\private\\contract.pdf"),
        )
        assert first.status == second.status == 201
        first_data = await first.json()
        second_data = await second.json()

    first_path = Path(first_data["path"])
    second_path = Path(second_data["path"])
    assert first_data["filename"] == second_data["filename"] == "contract.pdf"
    assert first_path != second_path
    assert first_path.read_bytes() == b"first"
    assert second_path.read_bytes() == b"second"


@pytest.mark.asyncio
async def test_upload_returns_docker_visible_path(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    adapter = _make_adapter()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/session-1/attachments",
            data=_file_form(b"docker payload"),
        )
        assert response.status == 201
        data = await response.json()

    assert data["path"].startswith("/root/.hermes/attachments/api_server/")
    host_files = list((hermes_home / "attachments" / "api_server").rglob("contract.pdf"))
    assert len(host_files) == 1
    assert host_files[0].read_bytes() == b"docker payload"


@pytest.mark.asyncio
async def test_url_upload_downloads_server_side_and_returns_same_metadata(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    adapter = _make_adapter()
    stored_path = hermes_home / "attachments" / "api_server" / "download.pdf"
    stored = session_attachments.StoredSessionAttachment(
        target=session_attachments.SessionAttachmentTarget(
            attachment_id="att_test",
            session_id="session-1",
            filename="download.pdf",
            content_type="application/pdf",
            host_path=stored_path,
            agent_path=str(stored_path),
        ),
        size=4,
        sha256=hashlib.sha256(b"data").hexdigest(),
    )
    download = AsyncMock(return_value=stored)
    monkeypatch.setattr(session_attachments, "store_session_attachment_from_url", download)

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/session-1/attachments/url",
            json={
                "url": "https://files.example.test/report.pdf",
                "filename": "download.pdf",
                "content_type": "application/pdf",
            },
        )
        assert response.status == 201
        data = await response.json()

    download.assert_awaited_once_with(
        "session-1",
        "https://files.example.test/report.pdf",
        filename="download.pdf",
        content_type="application/pdf",
    )
    assert data == {
        "object": "hermes.session.attachment",
        "id": "att_test",
        "session_id": "session-1",
        "filename": "download.pdf",
        "content_type": "application/pdf",
        "size": 4,
        "sha256": hashlib.sha256(b"data").hexdigest(),
        "path": str(stored_path),
    }


@pytest.mark.asyncio
async def test_upload_requires_existing_session_and_writes_nothing(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    adapter = _make_adapter()
    adapter._get_existing_session_or_404 = AsyncMock(
        return_value=(
            None,
            web.json_response(
                {"error": {"code": "session_not_found"}},
                status=404,
            ),
        )
    )

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/missing/attachments",
            data=_file_form(b"must not be written"),
        )
        assert response.status == 404

    assert not (hermes_home / "attachments").exists()


@pytest.mark.asyncio
async def test_upload_enforces_bearer_auth_before_session_lookup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    adapter = _make_adapter(api_key="secret-key")

    async with TestClient(TestServer(_create_app(adapter))) as client:
        denied = await client.post(
            "/api/sessions/session-1/attachments",
            data=_file_form(b"denied"),
        )
        allowed = await client.post(
            "/api/sessions/session-1/attachments",
            data=_file_form(b"allowed"),
            headers={"Authorization": "Bearer secret-key"},
        )

    assert denied.status == 401
    assert allowed.status == 201
    adapter._get_existing_session_or_404.assert_awaited_once_with("session-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_data", "expected_status", "expected_code"),
    [
        ({"file": "not multipart"}, 415, "unsupported_media_type"),
        (_file_form(b"wrong field", field_name="other"), 400, "missing_file"),
        (_file_form(b""), 400, "empty_attachment"),
    ],
)
async def test_upload_rejects_invalid_requests_and_cleans_partials(
    tmp_path,
    monkeypatch,
    request_data,
    expected_status,
    expected_code,
):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    adapter = _make_adapter()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/session-1/attachments",
            data=request_data,
        )
        assert response.status == expected_status
        data = await response.json()

    assert data["error"]["code"] == expected_code
    attachment_root = hermes_home / "attachments" / "api_server"
    assert not list(attachment_root.rglob("*.upload"))
    assert not [path for path in attachment_root.rglob("*") if path.is_file()]


@pytest.mark.asyncio
async def test_upload_rejects_oversize_file_and_cleans_partial(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(session_attachments, "MAX_SESSION_ATTACHMENT_BYTES", 8)
    adapter = _make_adapter()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/session-1/attachments",
            data=_file_form(b"123456789"),
        )
        assert response.status == 413
        data = await response.json()

    assert data["error"]["code"] == "attachment_too_large"
    attachment_root = hermes_home / "attachments" / "api_server"
    assert not list(attachment_root.rglob("*.upload"))
    assert not [path for path in attachment_root.rglob("*") if path.is_file()]


@pytest.mark.asyncio
async def test_upload_rejects_multipart_without_boundary(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    adapter = _make_adapter()

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/session-1/attachments",
            data=b"malformed multipart body",
            headers={"Content-Type": "multipart/form-data"},
        )
        assert response.status == 400
        data = await response.json()

    assert data["error"]["code"] == "invalid_multipart"
    assert not list((hermes_home / "attachments").rglob("*.upload"))


@pytest.mark.asyncio
async def test_upload_hides_internal_storage_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    adapter = _make_adapter()

    def _fail_allocation(*_args, **_kwargs):
        raise OSError("/private/host/path must not leak")

    monkeypatch.setattr(
        session_attachments,
        "allocate_session_attachment",
        _fail_allocation,
    )

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/session-1/attachments",
            data=_file_form(b"payload"),
        )
        assert response.status == 500
        raw_body = await response.text()

    assert "private/host/path" not in raw_body
    assert "attachment_upload_failed" in raw_body


def test_writer_cleans_partial_on_cancellation(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    target = session_attachments.allocate_session_attachment(
        "session-1",
        "cancelled.pdf",
        "application/pdf",
    )

    with pytest.raises(asyncio.CancelledError):
        with session_attachments.SessionAttachmentWriter(target) as writer:
            writer.write(b"partial")
            raise asyncio.CancelledError

    assert not target.host_path.exists()
    assert not target.host_path.parent.exists()
    assert not list(hermes_home.rglob("*.upload"))


def test_allocation_honors_context_local_profile_home(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope

    default_home = tmp_path / "default"
    worker_home = tmp_path / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")

    with _profile_runtime_scope(worker_home):
        target = session_attachments.allocate_session_attachment(
            "session-1",
            "profile.pdf",
            "application/pdf",
        )
        with session_attachments.SessionAttachmentWriter(target) as writer:
            writer.write(b"profile payload")
            writer.commit()

    assert target.host_path.is_relative_to(worker_home / "attachments")
    assert target.host_path.read_bytes() == b"profile payload"
    assert not (default_home / "attachments").exists()
