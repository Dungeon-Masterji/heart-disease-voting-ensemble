"""One consistent error format for every failure mode (400/422/429/503/500)."""
from typing import Optional


class ApiError(Exception):
    def __init__(self, status_code: int, error_code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


def error_body(error_code: str, detail: str, errors: Optional[list] = None) -> dict:
    body = {"error_code": error_code, "detail": detail}
    if errors:
        body["errors"] = errors
    return body
