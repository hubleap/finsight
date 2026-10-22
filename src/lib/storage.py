import base64
import os, re, io
from typing_extensions import Optional
from uuid import uuid4
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from docx import Document
import csv
import pandas as pd

from src.lib.config import UPLOAD_LIMIT, UPLOAD_PATH, UPLOAD_TYPES
from src.models.document import UploadDocument


class Storage:
    """Handles file upload, validation, and content extraction.

    Supports multiple file formats including PDF, Word, Excel, CSV,
    images, and plain text. Provides validation and binary-to-text
    conversion utilities.
    """

    def parse_data_url(self, url: str) -> Optional[str]:
        """Extract MIME type from a data URL.

        Args:
            url: Data URL in format data:<mime>;base64,<data>

        Returns:
            MIME type string or None if parsing fails
        """
        match = re.match(r"data:(?P<mime>[\w/-]+);base64,(?P<data>.+)", url)
        if not match:
            return None
        return match.group("mime")

    def upload(self, data: bytes, path: str, prefix: str, ext: str) -> str:
        """Upload binary data to the specified path.

        Args:
            data: Binary content to upload
            path: Destination directory path
            prefix: Filename prefix
            ext: File extension

        Returns:
            Full path to the uploaded file
        """
        path = path if path else UPLOAD_PATH
        os.makedirs(path, exist_ok=True)

        filename = f'{prefix if prefix else uuid4().hex}.{ext}'
        filepath = os.path.join(path, filename)

        with open(filepath, 'wb') as f:
            f.write(data)

        return filepath

    def binary_to_text(self, data: bytes, name: str, mime_type: Optional[str] = None) -> str:
        """Convert binary file content to text representation.

        Supports PDF, Word documents, Excel spreadsheets, CSV, plain text,
        and images (returned as base64 data URLs).

        Args:
            data: Binary file content
            name: Filename for error reporting
            mime_type: MIME type of the file

        Returns:
            Extracted text content or base64 data URL for images
        """
        try:
            if mime_type == 'application/pdf':
                pdf_reader = PdfReader(io.BytesIO(data))
                text = "\n\n".join(page.extract_text() or "" for page in pdf_reader.pages)
                return text.strip()

            elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                doc = Document(io.BytesIO(data))
                text = "\n\n".join(p.text or "" for p in doc.paragraphs)
                return text.strip()

            elif mime_type == 'text/csv':
                # decoded = data.decode('utf-8', errors='ignore')
                df = pd.read_csv(io.BytesIO(data))
                text = ""
                for _, row in df.iterrows():
                    text += " | ".join(f'{col}:{row[col]}' for col in df.columns)
                
                return text.strip()

            elif mime_type in ["text/plain", "text/markdown"]:
                text = data.decode('utf-8', errors='ignore')
                return text.strip()

            elif mime_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                try:
                    xls = pd.read_excel(io.BytesIO(data))
                    text = xls.to_csv(index=False)
                    return text.strip()
                except Exception as e:
                    return f"[Excel parse failed: {e}]"

            elif mime_type in ["image/png", "image/jpeg", "image/jpg"]:
                b64 = base64.b64encode(data).decode('utf-8')
                return f"data:{mime_type};base64,{b64}"

            else:
                return f"[Unsupported file type: {mime_type}]"

        except Exception as e:
            return f"[Error parsing {name}: {e}]"

    async def validate_upload(self, upload: UploadDocument) -> str:
        """Validate an uploaded document.

        Checks file type, extension, content presence, and file size
        against configured limits.

        Args:
            upload: UploadDocument model with file content

        Returns:
            File extension string

        Raises:
            HTTPException: 400 for invalid type/extension/empty file,
                          413 for file too large
        """
        if not upload.content_type or upload.content_type not in UPLOAD_TYPES:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Invalid file type. Allowed: {UPLOAD_TYPES}")

        filename = upload.filename or ''
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        allowed_exts = {'csv', 'pdf'}
        if ext not in allowed_exts:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Invalid file ext({ext}). Allowed: {allowed_exts}")

        if not upload.content:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f'File is empty: {upload.filename}')

        if len(upload.content) > UPLOAD_LIMIT:
            limit = round(UPLOAD_LIMIT / (1024 * 1024), 2)
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f'File too large: {upload.filename}, (max {limit}MB)')

        return ext

    def validate_image(self, content: bytes):
        """Validate that binary content is a valid image file.

        Args:
            content: Binary image content

        Raises:
            HTTPException: 400 if content is not a valid image
        """
        try:
            image = Image.open(io.BytesIO(content))
            image.load()
        except UnidentifiedImageError:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail='Invalid image file')

    def validate_csv(self, content: bytes):
        """Validate that binary content is a valid csv file.

        Args:
            content: Binary image content

        Raises:
            HTTPException: 400 if content is not a valid image
        """
        try:
            image = Image.open(io.BytesIO(content))
            image.load()
        except UnidentifiedImageError:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail='Invalid image file')

    def validate_pdf(self, content: bytes):
        """Validate that binary content is a valid PDF with extractable text.

        Args:
            content: Binary PDF content

        Raises:
            HTTPException: 400 if PDF is invalid or has no text
        """
        pdf_reader = PdfReader(io.BytesIO(content))
        text = "\n\n".join(page.extract_text() or "" for page in pdf_reader.pages).strip()
        if text == '':
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail='Invalid pdf file')

    async def upload_document(self, upload: UploadDocument, path: str = '', prefix: str = '') -> tuple:
        """Validate and upload a document.

        Performs type validation, format-specific validation for images
        and PDFs, then uploads the file.

        Args:
            upload: UploadDocument model with file content
            path: Destination directory path
            prefix: Filename prefix

        Returns:
            Tuple of (filepath, content_type)
        """
        ext = await self.validate_upload(upload)

        if upload.content_type == "text/csv":
            self.validate_image(upload.content)

        if upload.content_type == 'application/pdf':
            self.validate_pdf(upload.content)

        filepath = self.upload(upload.content, path, prefix, ext)
        return filepath, upload.content_type
    