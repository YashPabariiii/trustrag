from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    plan: str = Field(default="free", pattern="^(free|pro)$")


class RegisterResponse(BaseModel):
    tenant_id: UUID
    api_key: str
    plan: str
    # Shown once. Only the SHA-256 hash is stored.
    note: str = "Store this api_key now — it cannot be retrieved again."


class TokenRequest(BaseModel):
    api_key: str = Field(min_length=8)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
