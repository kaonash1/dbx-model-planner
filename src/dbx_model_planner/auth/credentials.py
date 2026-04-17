from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, repr=False)
class DatabricksCredentials:
    host: str
    token: str

    def masked_token(self) -> str:
        token = self.token
        if len(token) <= 8:
            return "***"
        return token[:4] + "..." + token[-4:]

    def __repr__(self) -> str:
        return f"DatabricksCredentials(host={self.host!r}, token={self.masked_token()!r})"


@dataclass(slots=True, repr=False)
class HuggingFaceCredentials:
    token: str | None = None

    @property
    def has_token(self) -> bool:
        return bool(self.token)

    def masked_token(self) -> str | None:
        if not self.token:
            return None
        token = self.token
        if len(token) <= 8:
            return "***"
        return token[:4] + "..." + token[-4:]

    def __repr__(self) -> str:
        return f"HuggingFaceCredentials(token={self.masked_token()!r})"
