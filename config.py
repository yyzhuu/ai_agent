from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


LLMProvider = Literal["openai"]
DBProvider = Literal["duckdb", "mysql"]
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

@dataclass(frozen=True)
class ModelConfig:
    provider: LLMProvider = field(
        default_factory=lambda: os.getenv("MODEL_PROVIDER", "openai").lower()
    )
    openai_model_name: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("MODEL_TEMPERATURE", "0.0"))
    )
    max_tokens: int | None = field(
        default_factory=lambda: int(os.getenv("MODEL_MAX_TOKENS", "2048"))
        if os.getenv("MODEL_MAX_TOKENS")
        else None
    )
    timeout: int = field(default_factory=lambda: int(os.getenv("MODEL_TIMEOUT", "120")))

    def validate(self) -> None:
        if self.provider != "openai":
            raise ValueError(f"Unsupported MODEL_PROVIDER={self.provider!r}. Use 'openai'.")


@dataclass(frozen=True)
class DatabaseConfig:
    provider: DBProvider = field(
        default_factory=lambda: os.getenv("DB_PROVIDER", "duckdb").lower()
    )

    duckdb_path: str = field(
        default_factory=lambda: os.getenv("DUCKDB_PATH", "securities.duckdb")
    )

    mysql_uri: str =f"mysql+pymysql://{USERNAME}:{PASSWORD}@host:3306/db_name"


    def validate(self) -> None:
        if self.provider not in {"duckdb", "mysql"}:
            raise ValueError(
                f"Unsupported DB_PROVIDER={self.provider!r}. Use 'duckdb' or 'mysql'."
            )

DEFAULT_DB_CONFIG = DatabaseConfig() 
DEFAULT_MODEL_CONFIG = ModelConfig() 
