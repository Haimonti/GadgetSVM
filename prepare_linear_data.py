"""Extract the official LIBSVM archives into the names used by src/config.py.

Download archives into data/downloads first; this script only extracts and
records hashes. Existing processed data is kept.
"""

import bz2
import hashlib
import json
import lzma
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOWNLOADS = ROOT / "data" / "downloads"
PROCESSED = ROOT / "data" / "processed"
FILES = {
    "gisette.binary.scale": "gisette_scale.bz2",
    "real-sim": "real-sim.bz2",
    "w8a": "w8a",
    "w8a.t": "w8a.t",
    "ijcnn1": "ijcnn1.bz2",
    "ijcnn1.t": "ijcnn1.t.bz2",
    "a9a": "a9a",
    "a9a.t": "a9a.t",
    "webspam_unigram.svm": "webspam_wc_normalized_unigram.svm.xz",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for target_name, source_name in FILES.items():
        target = PROCESSED / target_name
        source = DOWNLOADS / source_name
        if target.exists() and target.stat().st_size > 0:
            print(f"keeping {target}", flush=True)
            continue
        if not source.exists() or source.stat().st_size == 0:
            raise FileNotFoundError(source)
        temporary = target.with_name(target.name + ".part")
        opener = bz2.open if source.suffix == ".bz2" else (
            lzma.open if source.suffix == ".xz" else open)
        with opener(source, "rb") as inp, temporary.open("wb") as out:
            shutil.copyfileobj(inp, out, length=8 * 1024 * 1024)
        temporary.replace(target)
        print(f"extracted {target} ({target.stat().st_size:,} bytes)", flush=True)
    manifest = {}
    for path in sorted(PROCESSED.iterdir()):
        if path.is_file():
            manifest[path.name] = {"bytes": path.stat().st_size,
                                   "sha256": digest(path),
                                   "source_archive": FILES.get(path.name, "local source")}
    (ROOT / "linear_data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
