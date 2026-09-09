import sys
import bz2
import gzip
import shutil
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.datasets import dump_svmlight_file

from src.config import RAW_DIR, DATA_DIR

# UCI Covertype (the raw multiclass source). Downloaded, extracted, and
# preprocessed into a *binary* LIBSVM file so the pipeline can load it exactly
# like rcv1 / covtype.libsvm.binary.scale (see preprocess_covertype_uci).
COVERTYPE_URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
COVERTYPE_ZIP = "covertype.zip"
COVERTYPE_OUT = "covtype.uci.binary.scale"   # LIBSVM output written into DATA_DIR


def extract_bz2(src: Path, dest_dir: Path) -> Path:
    """Extract a single .bz2 file into dest_dir, named after the source stem."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.stem  # rcv1_train.binary.bz2 → rcv1_train.binary
    with bz2.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dest


def extract_all() -> dict[str, Path]:
    """
    Decompress every .bz2 in RAW_DIR flat into DATA_DIR, named after the
    archive stem. Already-extracted files are skipped (idempotent), so adding a
    new dataset does not force re-extracting the large existing archives.

    Examples:
        raw/rcv1_train.binary.bz2            -> processed/rcv1_train.binary
        raw/covtype.libsvm.binary.scale.bz2  -> processed/covtype.libsvm.binary.scale
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    archives = sorted(RAW_DIR.glob("*.bz2"))
    if not archives:
        raise FileNotFoundError(f"No .bz2 files found in {RAW_DIR}")

    paths = {}
    for src in archives:
        dest = DATA_DIR / src.stem  # flat output, e.g. covtype.libsvm.binary.scale
        if dest.exists():
            print(f"[{src.name}] already extracted -> {dest} (skipped)")
        else:
            dest = extract_bz2(src, DATA_DIR)
            print(f"[{src.name}] -> {dest}")
        paths[src.stem] = dest
    return paths


# --------------------------------------------------------------------------- #
# UCI Covertype: download -> unzip -> gunzip -> preprocess -> LIBSVM           #
# --------------------------------------------------------------------------- #
def download_file(url: str, dest: Path) -> Path:
    """Download `url` to `dest` (idempotent — skips if the file already exists)."""
    if dest.exists():
        print(f"[{dest.name}] already downloaded -> {dest} (skipped)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    return dest


def extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract every member of `zip_path` into `dest_dir`; return the member paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
        return [dest_dir / name for name in zf.namelist()]


def extract_gz(src: Path, dest: Path) -> Path:
    """Decompress a single .gz file `src` to `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dest


def preprocess_covertype_uci() -> Path:
    """Download the UCI Covertype zip and turn it into a binary LIBSVM file.

    Steps (all idempotent):
      1. download covertype.zip into RAW_DIR
      2. unzip it (it contains covtype.data.gz plus info files)
      3. gunzip covtype.data.gz -> covtype.data (CSV: 581,012 rows x 55 cols =
         54 features + Cover_Type label in 1..7)
      4. binarise: the majority class 2 (Lodgepole Pine) vs. the rest -> +/-1,
         the same "class 2 vs rest" task as the ready-made covtype.binary
      5. min-max scale every feature to [0, 1] (the 44 binary columns stay 0/1)
      6. write DATA_DIR/covtype.uci.binary.scale in LIBSVM format, which
         load_svmlight_file reads exactly like rcv1 / covtype.libsvm.binary.scale
    """
    out_path = DATA_DIR / COVERTYPE_OUT
    if out_path.exists():
        print(f"[{COVERTYPE_OUT}] already built -> {out_path} (skipped)")
        return out_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1-2. download + unzip
    zip_path = download_file(COVERTYPE_URL, RAW_DIR / COVERTYPE_ZIP)
    extract_dir = RAW_DIR / "covertype_uci"
    members = extract_zip(zip_path, extract_dir)
    gz = next((p for p in members if p.name == "covtype.data.gz"), None)
    if gz is None:  # fall back to any *.data.gz inside the archive
        gz = next(extract_dir.rglob("*.data.gz"), None)
    if gz is None:
        raise FileNotFoundError(f"covtype.data.gz not found inside {zip_path}")

    # 3. gunzip the CSV
    csv_path = extract_dir / "covtype.data"
    if not csv_path.exists():
        extract_gz(gz, csv_path)
    print(f"[extract] {gz.name} -> {csv_path}")

    # 4. load (pandas if available — much faster; else numpy)
    print(f"[preprocess] loading {csv_path} ...")
    try:
        import pandas as pd
        data = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.float64)
    except ImportError:
        data = np.loadtxt(csv_path, delimiter=",")

    X = data[:, :54]
    cover_type = data[:, 54].astype(int)

    # 5. binarise: majority class (2) vs. the rest
    y = np.where(cover_type == 2, 1.0, -1.0)

    # 6. min-max scale each feature to [0, 1] (binary columns stay 0/1)
    col_min = X.min(axis=0)
    col_range = X.max(axis=0) - col_min
    col_range[col_range == 0] = 1.0
    X = (X - col_min) / col_range

    # 7. write LIBSVM (sparse; zeros dropped), 1-based indices as LIBSVM expects
    dump_svmlight_file(X, y, str(out_path), zero_based=False)
    print(f"[preprocess] wrote {out_path}  ({X.shape[0]} samples, "
          f"{X.shape[1]} features, +{int((y > 0).sum())} / -{int((y < 0).sum())})")
    return out_path


if __name__ == "__main__":
    # bz2 archives (rcv1, ready-made covtype) — only if any are present
    if sorted(RAW_DIR.glob("*.bz2")):
        extracted = extract_all()
        print("\nExtracted archives:")
        for name, path in extracted.items():
            print(f"  {name}: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"No .bz2 archives in {RAW_DIR} (skipping bz2 extraction).")

    # UCI Covertype — download + preprocess
    print("\nBuilding UCI Covertype (download + preprocess) ...")
    covtype_uci = preprocess_covertype_uci()
    print(f"  covertype (UCI): {covtype_uci}  ({covtype_uci.stat().st_size / 1e6:.1f} MB)")
    print("\nDone.")
