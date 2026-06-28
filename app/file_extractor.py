
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from pypdf import PdfReader
from io import BytesIO
from pypdf.errors import PdfReadError

from fastapi import HTTPException, UploadFile


@dataclass
class ExtractedFile:
    text: str
    metadata: dict[str, Any]


def extract_text(content: bytes) -> ExtractedFile:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be valid UTF-8 text"
        )

    return ExtractedFile(
        text=text,
        metadata={}
    )

def extract_pdf(content: bytes) -> ExtractedFile:
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise HTTPException(status_code=400, detail="Encrypted PDFs are not supported")

        pages_text = []
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text and extracted_text.strip():
                pages_text.append(extracted_text)

        if not pages_text:
            raise HTTPException(
                status_code=400,
                detail="PDF contains no extractable text",
            )

        return ExtractedFile(text="\n\n".join(pages_text), metadata={"page_count": len(reader.pages)})
    except PdfReadError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF")


EXTRACTORS = {
    ".md": extract_text,
    ".txt": extract_text,
    ".pdf": extract_pdf
}


async def extract_upload_file(file: UploadFile) -> ExtractedFile:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    extension = Path(file.filename).suffix.lower()
    extractor = EXTRACTORS.get(extension)
    if extractor is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Supported types: {list(EXTRACTORS.keys())}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    extracted = extractor(content)
    metadata = {
        "source": "file",
        "filename": file.filename,
        "content_type": file.content_type,
        "extension": extension,
        **extracted.metadata,
    }
    return ExtractedFile(
        text=extracted.text,
        metadata=metadata
    )