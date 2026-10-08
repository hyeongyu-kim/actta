"""Verify the release's code, configurations, references, and result checksums."""

import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "results/publication_manifest.json").read_text())
    failures = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            failures.append(f"Missing: {entry['path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(f"SHA-256 mismatch: {entry['path']}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Verified {len(manifest['files'])} publication file checksums.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
