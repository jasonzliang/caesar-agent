"""Native TXT save/load must not mistake generated headings for new sections."""
from __future__ import annotations

import ast
import json
import os
import socket
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("TXT parser tests must not make network calls")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.fixture
def save_native(tmp_path):
    # Execute the production serializer unchanged, without importing provider,
    # KB or agent modules or constructing any of their clients.
    source = Path(__file__).resolve().parents[3] / "caesar" / "artifact_synthesis.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "ArtifactSynthesizer")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in {"_save_synthesis", "_format_source_list"}]
    namespace = {"Dict": dict, "datetime": datetime, "os": os,
                 "json": json, "SYNTHESIS_SAVE_JSON": False}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    writer = namespace["ArtifactSynthesizer"]()
    writer.agent = SimpleNamespace(get_id=lambda: "fixture")
    writer.logger = SimpleNamespace(error=pytest.fail)
    writer.saved_artifact_files = []

    def save(artifact, *, abstract="Overview [9].", sources=None):
        result = {"abstract": abstract, "artifact": artifact,
                  "sources": sources if sources is not None else {
                      "https://example.test/first": 2,
                      "https://example.test/second": 9,
                  }}
        writer._save_synthesis(result, base_dir=str(tmp_path), suffix="merged-3",
                               timestamp="11060000")
        path = tmp_path / "fixture.merged-3.11060000.txt"
        return path, result

    return save


@pytest.mark.parametrize("header", ["SOURCES:", "ARTIFACT:", "ABSTRACT:"])
@pytest.mark.parametrize("layout", ["{header}\nLiteral heading.",
                                    "```text\n{header}\nLiteral heading.\n```",
                                    "~~~text\n{header}\nLiteral heading.\n~~~",
                                    "    {header}\n    Literal heading."])
@pytest.mark.parametrize("with_sources", [True, False])
def test_native_roundtrip_preserves_headers_inside_main_text(save_native, header, layout,
                                                            with_sources):
    from app.routers.artifacts import _parse_synthesis_txt

    body = "First claim [2].\n\n" + layout.format(header=header) + "\n\nLater claim [9]."
    path, expected = save_native(body, sources=None if with_sources else {})
    parsed = _parse_synthesis_txt(path.read_text(encoding="utf-8"))
    assert parsed["artifact"] == expected["artifact"]
    assert parsed["abstract"] == expected["abstract"]
    assert parsed["sources"] == expected["sources"]


@pytest.mark.parametrize("abstract", ["", "A summary [2].\n\nSOURCES:\nSummary detail [9].",
                                     "A summary [2].\n\nABSTRACT:\nSummary detail [9]."])
def test_native_roundtrip_preserves_optional_abstract(save_native, abstract):
    from app.routers.artifacts import _parse_synthesis_txt

    path, expected = save_native("Body [2,9]. Unknown [77].", abstract=abstract)
    parsed = _parse_synthesis_txt(path.read_text(encoding="utf-8"))
    assert parsed["abstract"] == expected["abstract"]
    assert parsed["artifact"] == expected["artifact"]
    assert parsed["sources"] == expected["sources"]


@pytest.mark.parametrize("tail", ["SOURCES:", "SOURCES:\nNot a source row [9].",
                                  "SOURCES:\n[7] https://example.test/sample\n"
                                  "Prose continues [9]."])
def test_non_footer_sources_section_stays_in_answer(save_native, tail):
    from app.routers.artifacts import _parse_synthesis_txt

    body = "Claim [2].\n\n" + tail
    path, _ = save_native(body, sources={})
    parsed = _parse_synthesis_txt(path.read_text(encoding="utf-8"))
    assert parsed["artifact"] == body
    assert parsed["sources"] == {}


def test_source_rows_keep_sparse_ids_and_full_urls(save_native):
    from app.routers.artifacts import _parse_synthesis_txt

    sources = {"https://example.test/report (draft)?x=[2]": 9,
               "https://example.test/uncited": 77,
               "https://example.test/first": 2}
    path, _ = save_native("Claim [2,9]. `array[77]`.", sources=sources)
    parsed = _parse_synthesis_txt(path.read_text(encoding="utf-8"))
    assert parsed["sources"] == sources
    assert parsed["artifact"] == "Claim [2,9]. `array[77]`."


def test_legacy_case_whitespace_crlf_and_blank_source_lines():
    from app.routers.artifacts import _parse_synthesis_txt

    text = ("\r\n abstract: \r\nSummary [2].\r\n\r\n artifact: \r\nClaim [2].\r\n\r\n"
            " sources: \r\n [2] https://example.test/a\r\n\r\n [9] https://example.test/b\r\n")
    parsed = _parse_synthesis_txt(text)
    assert parsed["abstract"] == "Summary [2]."
    assert parsed["artifact"] == "Claim [2]."
    assert parsed["sources"] == {"https://example.test/a": 2, "https://example.test/b": 9}


def test_code_indentation_is_not_stripped_from_main_text(save_native):
    from app.routers.artifacts import _parse_synthesis_txt

    body = "    array[2]\n    value = 2 * 3\n\nClaim [9]."
    path, _ = save_native(body, abstract="")
    parsed = _parse_synthesis_txt(path.read_text(encoding="utf-8"))
    assert parsed["artifact"] == body


def test_ambiguous_abstract_delimiter_never_discards_remaining_text(save_native):
    from app.routers.artifacts import _parse_synthesis_txt

    # The unescaped legacy format cannot distinguish this abstract heading
    # from the real delimiter. Taking the first boundary must retain all later
    # prose (including the real delimiter) rather than resetting the body.
    path, _ = save_native("Actual body [9].",
                          abstract="Summary [2].\n\nARTIFACT:\nMore summary [9].")
    parsed = _parse_synthesis_txt(path.read_text(encoding="utf-8"))
    assert parsed["abstract"] == "Summary [2]."
    assert parsed["artifact"] == "More summary [9].\n\nARTIFACT:\nActual body [9]."


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["SOURCES:", "ARTIFACT:", "ABSTRACT:"])
async def test_synthesis_api_returns_all_native_body_and_source_rows(client, save_native,
                                                                    header):
    from app.db import SessionLocal
    from app.models import Run

    body = f"First claim [2].\n\n```text\n{header}\nExample\n```\n\nLast claim [9]."
    path, expected = save_native(body)
    async with SessionLocal() as session:
        session.add(Run(id="native-fixture", query="Fixture", preset="fast", status="completed",
                        repository=str(path.parent)))
        await session.commit()
    response = await client.get("/runs/native-fixture/synthesis", params={"draft": "latest"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["artifact"] == expected["artifact"]
    assert data["abstract"] == expected["abstract"]
    assert data["sources"] == [{"index": 2, "url": "https://example.test/first"},
                               {"index": 9, "url": "https://example.test/second"}]


def test_indented_source_example_at_end_is_not_a_footer(save_native):
    from app.routers.artifacts import _parse_synthesis_txt

    body = "Claim [2].\n\n    SOURCES:\n    [9] https://example.test/literal"
    path, _ = save_native(body, sources={})
    parsed = _parse_synthesis_txt(path.read_text(encoding="utf-8"))
    assert parsed["artifact"] == body
    assert parsed["sources"] == {}


def test_legacy_exact_footer_shape_cannot_identify_its_original_author(save_native):
    # Native sources and identical model-written final lines serialize to the
    # same bytes. No TXT-only parser can recover which producer wrote them.
    path, _ = save_native("Claim [2].", abstract="", sources={"https://example.test/a": 2})
    native = path.read_text(encoding="utf-8")
    path, _ = save_native("Claim [2].\n\nSOURCES:\n[2] https://example.test/a",
                          abstract="", sources={})
    assert path.read_text(encoding="utf-8") == native
