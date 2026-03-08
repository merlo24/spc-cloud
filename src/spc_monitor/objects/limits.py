from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Limits:
    lcl: Optional[float]
    ucl: Optional[float]
    cl: Optional[float] = None
    meta: dict[str, Any] | None = None