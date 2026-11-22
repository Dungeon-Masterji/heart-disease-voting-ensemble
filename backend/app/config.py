"""Runtime settings, read from environment variables with safe local defaults."""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "training" / "artifacts" / "models"


@dataclass(frozen=True)
class Settings:
    model_path: Path = DEFAULT_MODEL_DIR / "model_m2.0.0.joblib"
    meta_path: Path = DEFAULT_MODEL_DIR / "model_m2.0.0.json"
    # Second ensemble, for patients without ca (fluoroscopy) and/or thal (thallium scan).
    noscan_model_path: Path = DEFAULT_MODEL_DIR / "model_m2.0.0-noscan.joblib"
    noscan_meta_path: Path = DEFAULT_MODEL_DIR / "model_m2.0.0-noscan.json"
    model_card_path: Path = REPO_ROOT / "docs" / "MODEL_CARD.md"
    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")
    rate_limit_per_minute: int = 60  # 0 disables rate limiting

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        origins = os.getenv("CORS_ORIGINS")
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", str(d.model_path))),
            meta_path=Path(os.getenv("META_PATH", str(d.meta_path))),
            noscan_model_path=Path(
                os.getenv("NOSCAN_MODEL_PATH", str(d.noscan_model_path))
            ),
            noscan_meta_path=Path(
                os.getenv("NOSCAN_META_PATH", str(d.noscan_meta_path))
            ),
            model_card_path=Path(os.getenv("MODEL_CARD_PATH", str(d.model_card_path))),
            cors_origins=(
                tuple(o.strip() for o in origins.split(",") if o.strip())
                if origins
                else d.cors_origins
            ),
            rate_limit_per_minute=int(
                os.getenv("RATE_LIMIT_PER_MINUTE", str(d.rate_limit_per_minute))
            ),
        )
