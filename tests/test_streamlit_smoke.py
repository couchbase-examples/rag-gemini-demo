"""No-secret smoke test: boot the Streamlit app and drive it with Playwright."""

import re

import pytest
from playwright.sync_api import Page, expect, sync_playwright

PDF_TEXT = "Couchbase smoke test document"


def _make_pdf(text):
    """Build a one-page PDF containing `text` (no extra dependencies)."""
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


@pytest.fixture(scope="module")
def page(streamlit_server):
    url, _ = streamlit_server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_default_timeout(30_000)
        page.goto(url)
        yield page
        browser.close()


def _assert_no_errors(page: Page, log_path):
    expect(page.get_by_test_id("stException")).to_have_count(0)
    assert "Traceback" not in page.inner_text("body")
    log = log_path.read_text()
    assert "Traceback" not in log, log


def test_app_renders(page: Page, streamlit_server):
    _, log_path = streamlit_server
    expect(page.get_by_role("heading", name="Chat with PDF")).to_be_visible()
    expect(page.get_by_text("Hi, I'm a chatbot who can chat with the PDF")).to_be_visible()
    expect(page.get_by_placeholder("Ask a question based on the PDF")).to_be_visible()
    sidebar = page.get_by_test_id("stSidebar")
    expect(sidebar.get_by_text("Upload your PDF")).to_be_visible()
    expect(sidebar.get_by_role("button", name="Upload")).to_be_visible()
    expect(sidebar.get_by_text("How does it work?")).to_be_visible()
    _assert_no_errors(page, log_path)


def test_pdf_upload(page: Page, streamlit_server):
    _, log_path = streamlit_server
    sidebar = page.get_by_test_id("stSidebar")
    sidebar.locator("input[type=file]").set_input_files(
        {"name": "smoke.pdf", "mimeType": "application/pdf", "buffer": _make_pdf(PDF_TEXT)}
    )
    expect(sidebar.get_by_text("smoke.pdf")).to_be_visible()
    sidebar.get_by_role("button", name="Upload").click()
    expect(page.get_by_text("PDF loaded into vector store in 1 documents")).to_be_visible()
    _assert_no_errors(page, log_path)


def test_chat_round_trip(page: Page, streamlit_server):
    _, log_path = streamlit_server
    chat = page.get_by_placeholder("Ask a question based on the PDF")
    chat.fill("What is this document about?")
    chat.press("Enter")
    expect(page.get_by_text("What is this document about?")).to_be_visible()
    # One answer from the RAG chain, one from the plain LLM chain.
    expect(page.get_by_text(re.compile(r"^Smoke answer from models/"))).to_have_count(2)
    _assert_no_errors(page, log_path)
