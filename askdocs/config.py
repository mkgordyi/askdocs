"""Settings. Everything is read from environment variables with safe defaults."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_PROVIDER = os.getenv("ASKDOCS_LLM_PROVIDER", "none")
# Without an LLM the similarity threshold is the only judge, so it is strict (calibrated by
# evals/run.py --calibrate). With an LLM it is only a cheap pre-filter: the model then decides
# from the passages whether the question can be answered.
_DEFAULT_MIN_CONFIDENCE = "0.64" if _PROVIDER == "none" else "0.55"


@dataclass(frozen=True)
class Settings:
    source_file: Path = ROOT / "data" / "source" / "eu-ai-act-2024-1689-en.xhtml"
    index_dir: Path = Path(os.getenv("ASKDOCS_INDEX_DIR", ROOT / "index"))
    model_cache: Path = Path(os.getenv("ASKDOCS_MODEL_CACHE", ROOT / ".models"))
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    top_k: int = int(os.getenv("ASKDOCS_TOP_K", "5"))
    # Below this similarity the service refuses to answer.
    min_confidence: float = float(os.getenv("ASKDOCS_MIN_CONFIDENCE", _DEFAULT_MIN_CONFIDENCE))

    # "none" = quote the best passage; "openai" or "azure_openai" = write an answer from passages.
    llm_provider: str = _PROVIDER
    llm_model: str = os.getenv("ASKDOCS_LLM_MODEL", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    price_in_per_1m: float = float(os.getenv("ASKDOCS_PRICE_IN_PER_1M", "0"))
    price_out_per_1m: float = float(os.getenv("ASKDOCS_PRICE_OUT_PER_1M", "0"))


settings = Settings()
