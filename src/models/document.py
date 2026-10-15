from typing import Optional

from pydantic import BaseModel, Field


class UploadDocument(BaseModel):
    """Model representing an uploaded document.

    Attributes:
        url: Base64-encoded data URL of the document
        content_type: MIME type of the document
        filename: Original filename of the uploaded document
        content: Raw binary content of the document
    """
    url: str = Field(..., description='The base64 url of the document')
    content_type: Optional[str] = Field(None, description='The content type of the document')
    filename: Optional[str] = Field(None, description='The name of the document')
    content: Optional[bytes] = Field(None, description='The bytes content')
