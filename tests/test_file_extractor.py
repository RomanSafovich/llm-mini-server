
from io import BytesIO
from pathlib import Path
from fastapi import HTTPException, UploadFile
from pypdf import PdfWriter
import pytest

from app.file_extractor import extract_pdf, extract_upload_file


@pytest.mark.parametrize(
    "filename, content, expected_extension",
    [
        ("test.md", b"# Hello\nThis is markdown.", ".md"),
        ("test.txt", b"Hello\nThis is text.", ".txt"),
    ],
)
@pytest.mark.anyio
async def test_extractor_text_files(filename, content, expected_extension):
    file = UploadFile(
        filename=filename,
        file=BytesIO(content)
    )

    res = await extract_upload_file(file)
    assert res.text == content.decode("utf-8")
    assert res.metadata["filename"] == filename
    assert res.metadata["extension"] == expected_extension


@pytest.mark.anyio
async def test_extractor_unsupported_extension():
    file = UploadFile(
        filename="test.tst",
        file=BytesIO(b"Hello\nThis is text.")
    )
    with pytest.raises(HTTPException) as exc:
        await extract_upload_file(file)

    assert exc.value.status_code == 400
    assert "Unsupported file type" in exc.value.detail


@pytest.mark.anyio
async def test_extractor_empty_file():
    file = UploadFile(
        filename="test.md",
        file=BytesIO()
    )
    with pytest.raises(HTTPException) as exc:
        await extract_upload_file(file)

    assert exc.value.status_code == 400
    assert "Uploaded file is empty" in exc.value.detail


@pytest.mark.anyio
async def test_extractor_invalid_utf8():
    file = UploadFile(
        filename="test.md",
        file=BytesIO(b"\xff\xfe\xfa")
    )
    with pytest.raises(HTTPException) as exc:
        await extract_upload_file(file)

    assert exc.value.status_code == 400
    assert "File must be valid UTF-8 text" in exc.value.detail


def make_pdf_bytes(is_encrypted: bool = False) -> bytes:
    buffer = BytesIO()

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    if is_encrypted:
        writer.encrypt("test-password")

    writer.write(buffer)
    return buffer.getvalue()


def test_extract_pdf_rejects_encrypted_pdf():
    content = make_pdf_bytes(is_encrypted=True)

    with pytest.raises(HTTPException) as exc:
        extract_pdf(content)

    assert exc.value.status_code == 400
    assert exc.value.detail == "Encrypted PDFs are not supported"


@pytest.mark.anyio
async def test_corrupted_pdf():
    file = UploadFile(
        filename="test.pdf",
        file=BytesIO(b"This is not a real PDF.")
    )

    with pytest.raises(HTTPException) as exc:
        await extract_upload_file(file)

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid or corrupted PDF"


@pytest.mark.anyio
async def test_extractor_valid_pdf():
    fixture_path = Path(__file__).parent / "fixtures" / "sample.pdf"
    content = fixture_path.read_bytes()

    file = UploadFile(
        filename="sample.pdf",
        file=BytesIO(content),
    )

    result = await extract_upload_file(file)

    assert "Hello from PDF fixture" in result.text
    assert result.metadata["extension"] == ".pdf"
    assert result.metadata["page_count"] == 1


@pytest.mark.anyio
async def test_no_extractable_text_pdf():
    content = make_pdf_bytes()
    file = UploadFile(
        filename="test.pdf",
        file=BytesIO(content)
    )

    with pytest.raises(HTTPException) as exc:
        await extract_upload_file(file)

    assert exc.value.status_code == 400
    assert exc.value.detail == "PDF contains no extractable text"


@pytest.mark.anyio
async def test_extractor_missing_filename():
    file = UploadFile(
        filename="",
        file=BytesIO(b"Hello\nThis is text.")
    )
    with pytest.raises(HTTPException) as exc:
        await extract_upload_file(file)

    assert exc.value.status_code == 400
    assert "Filename is required" in exc.value.detail