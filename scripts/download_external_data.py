"""Download the three UCI hospitals used only for external validation.

Usage:  python scripts/download_external_data.py
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
EXPECTED_ROWS = {"hungarian": 294, "switzerland": 123, "va": 200}
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for site, expected in EXPECTED_ROWS.items():
        path = OUT_DIR / f"processed.{site}.data"
        if not path.exists():
            url = BASE_URL + path.name
            try:
                urllib.request.urlretrieve(url, path)  # nosec B310
            except urllib.error.URLError as exc:
                print(f"Download failed: {exc}", file=sys.stderr)
                print(
                    "If this is a certificate error on macOS, run Python's 'Install "
                    f"Certificates.command', or: curl -o {path} {url}",
                    file=sys.stderr,
                )
                return 1
        rows = [line for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != expected:
            path.unlink()
            print(f"{site}: expected {expected} rows, got {len(rows)}", file=sys.stderr)
            return 1
        print(f"{site}: {len(rows)} patients")
    return 0


if __name__ == "__main__":
    sys.exit(main())
