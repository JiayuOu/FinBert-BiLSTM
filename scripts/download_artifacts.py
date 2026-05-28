from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from check_artifacts import main as check_artifacts


DEFAULT_REPO_ID = "KallonOu/FinBert-BiLSTM-artifacts"
DEFAULT_FILENAME = "project_artifacts.zip"


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and extract project artifacts from Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Artifact zip filename in the dataset repo.")
    parser.add_argument("--local-dir", default=".", help="Repository root where files should be placed.")
    parser.add_argument("--no-extract", action="store_true", help="Only download the zip; do not extract it.")
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="dataset",
            local_dir=local_dir,
        )
    )
    print(f"Downloaded {zip_path}")

    if not args.no_extract:
        print(f"Extracting to {local_dir}")
        extract_zip(zip_path, local_dir)
        return check_artifacts()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
