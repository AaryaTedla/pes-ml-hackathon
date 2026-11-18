"""Runtime settings. Relative paths resolve from the repository, not the shell."""
from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    trips_dir: Path
    artifacts_dir: Path
    mode: str = "nearest"
    cors_origins: tuple[str, ...] = ("*",)

    @classmethod
    def from_env(cls):
        mode = os.getenv("PATHSNAP_MODE", "nearest").strip().lower()
        if mode not in {"nearest", "classifier", "hmm"}:
            raise ValueError("PATHSNAP_MODE must be nearest, classifier or hmm")
        origins = tuple(x.strip() for x in os.getenv("PATHSNAP_CORS_ORIGINS", "*").split(",") if x.strip())
        return cls(repo_path(os.getenv("PATHSNAP_TRIPS_DIR", "data/student_test")),
                   repo_path(os.getenv("PATHSNAP_ARTIFACTS_DIR", "artifacts")), mode, origins)
