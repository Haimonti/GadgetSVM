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

# Raw-data download links (kept here so `python data/extract_data.py` fetches and
# preprocesses every dataset into a binary LIBSVM file the pipeline can load).
# After a dataset is built, its downloaded archive + temp extraction files are
# deleted — only the preprocessed file in DATA_DIR is kept.
COVERTYPE_URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
COVERTYPE_ZIP = "covertype.zip"
COVERTYPE_OUT = "covtype.uci.binary.scale"

GISETTE_URL = "https://archive.ics.uci.edu/static/public/170/gisette.zip"
GISETTE_ZIP = "gisette.zip"
GISETTE_OUT = "gisette.binary.scale"

REALSIM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2"
REALSIM_BZ2 = "real-sim.bz2"
REALSIM_OUT = "real-sim"


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
# Shared download / archive / cleanup helpers                                 #
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


def _cleanup(*paths: Path) -> None:
    """Remove downloaded archives / temp extraction dirs once the preprocessed
    LIBSVM file is written — keep only the processed data, drop the redundant raw."""
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                print(f"[cleanup] removed dir  {p}")
            elif p.exists():
                p.unlink()
                print(f"[cleanup] removed file {p}")
        except OSError:
            pass


def _min_max_scale(X: np.ndarray) -> np.ndarray:
    """Scale every column to [0, 1]; binary (0/1) columns are left unchanged."""
    col_min = X.min(axis=0)
    col_range = X.max(axis=0) - col_min
    col_range[col_range == 0] = 1.0
    return (X - col_min) / col_range


# --------------------------------------------------------------------------- #
# UCI Covertype: download -> unzip -> gunzip -> preprocess -> LIBSVM           #
# --------------------------------------------------------------------------- #
def preprocess_covertype_uci() -> Path:
    """Download the UCI Covertype zip and turn it into a binary LIBSVM file.

    Steps (all idempotent):
      1. download covertype.zip into RAW_DIR
      2. unzip it (it contains covtype.data.gz plus info files)
      3. gunzip covtype.data.gz -> covtype.data (CSV: 581,012 rows x 55 cols =
         54 features + Cover_Type label in 1..7)
      4. binarise: the majority class 2 (Lodgepole Pine) vs. the rest -> +/-1
      5. min-max scale every feature to [0, 1] (the 44 binary columns stay 0/1)
      6. write DATA_DIR/covtype.uci.binary.scale in LIBSVM format
      7. delete the downloaded zip + temp extraction dir (keep only the output)
    """
    out_path = DATA_DIR / COVERTYPE_OUT
    if out_path.exists():
        print(f"[{COVERTYPE_OUT}] already built -> {out_path} (skipped)")
        return out_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = download_file(COVERTYPE_URL, RAW_DIR / COVERTYPE_ZIP)
    extract_dir = RAW_DIR / "covertype_uci"
    members = extract_zip(zip_path, extract_dir)
    gz = next((p for p in members if p.name == "covtype.data.gz"), None)
    if gz is None:
        gz = next(extract_dir.rglob("*.data.gz"), None)
    if gz is None:
        raise FileNotFoundError(f"covtype.data.gz not found inside {zip_path}")

    csv_path = extract_dir / "covtype.data"
    if not csv_path.exists():
        extract_gz(gz, csv_path)
    print(f"[extract] {gz.name} -> {csv_path}")

    print(f"[preprocess] loading {csv_path} ...")
    try:
        import pandas as pd
        data = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.float64)
    except ImportError:
        data = np.loadtxt(csv_path, delimiter=",")

    X = data[:, :54]
    cover_type = data[:, 54].astype(int)
    y = np.where(cover_type == 2, 1.0, -1.0)     # majority class vs. the rest
    X = _min_max_scale(X)

    dump_svmlight_file(X, y, str(out_path), zero_based=False)
    print(f"[preprocess] wrote {out_path}  ({X.shape[0]} samples, "
          f"{X.shape[1]} features, +{int((y > 0).sum())} / -{int((y < 0).sum())})")
    _cleanup(zip_path, extract_dir)
    return out_path


# --------------------------------------------------------------------------- #
# Gisette (UCI): download -> unzip -> parse dense .data/.labels -> LIBSVM      #
# --------------------------------------------------------------------------- #
def _load_dense(path: Path) -> np.ndarray:
    """Load a whitespace-separated dense numeric matrix (pandas if available)."""
    try:
        import pandas as pd
        return pd.read_csv(path, sep=r"\s+", header=None).to_numpy(dtype=np.float64)
    except ImportError:
        return np.loadtxt(path)


def preprocess_gisette() -> Path:
    """Download the UCI Gisette zip and turn it into a binary LIBSVM file.

    The zip ships the NIPS-2003 layout: dense `gisette_train.data` (6000 x 5000,
    whitespace-separated) and `gisette_train.labels` (+/-1). Steps (idempotent):
      1. download gisette.zip into RAW_DIR and unzip it
      2. locate *train.data and *train.labels
      3. min-max scale each feature to [0, 1]
      4. write DATA_DIR/gisette.binary.scale in LIBSVM format
      5. delete the downloaded zip + temp extraction dir (keep only the output)
    """
    out_path = DATA_DIR / GISETTE_OUT
    if out_path.exists():
        print(f"[{GISETTE_OUT}] already built -> {out_path} (skipped)")
        return out_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = download_file(GISETTE_URL, RAW_DIR / GISETTE_ZIP)
    extract_dir = RAW_DIR / "gisette"
    extract_zip(zip_path, extract_dir)

    data_file = next(extract_dir.rglob("*train.data"), None)
    label_file = next(extract_dir.rglob("*train.labels"), None)
    if data_file is None or label_file is None:
        raise FileNotFoundError(
            f"gisette train .data/.labels not found under {extract_dir}. "
            "Adjust preprocess_gisette() to the zip's actual layout."
        )

    print(f"[preprocess] loading {data_file.name} + {label_file.name} ...")
    X = _load_dense(data_file)
    y = np.sign(np.loadtxt(label_file)).astype(np.float64)   # already +/-1
    X = _min_max_scale(X)

    dump_svmlight_file(X, y, str(out_path), zero_based=False)
    print(f"[preprocess] wrote {out_path}  ({X.shape[0]} samples, "
          f"{X.shape[1]} features, +{int((y > 0).sum())} / -{int((y < 0).sum())})")
    _cleanup(zip_path, extract_dir)
    return out_path


# --------------------------------------------------------------------------- #
# real-sim (LibSVM): download .bz2 -> decompress -> LIBSVM (already scaled)    #
# --------------------------------------------------------------------------- #
def preprocess_real_sim() -> Path:
    """Download the LibSVM real-sim archive and decompress it.

    real-sim is already a single LIBSVM file (72,309 examples, 20,958 features,
    +/-1 labels), so it only needs downloading + decompressing; the loader splits
    it into train/test. The downloaded .bz2 is deleted afterwards. Idempotent.
    """
    out_path = DATA_DIR / REALSIM_OUT
    if out_path.exists():
        print(f"[{REALSIM_OUT}] already built -> {out_path} (skipped)")
        return out_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    bz2_path = download_file(REALSIM_URL, RAW_DIR / REALSIM_BZ2)
    dest = extract_bz2(bz2_path, DATA_DIR)          # -> DATA_DIR/real-sim
    print(f"[extract] {bz2_path.name} -> {dest}")
    _cleanup(bz2_path)
    return dest


if __name__ == "__main__":
    # bz2 archives (rcv1, ready-made covtype) — only if any are present
    if sorted(RAW_DIR.glob("*.bz2")):
        extracted = extract_all()
        print("\nExtracted archives:")
        for name, path in extracted.items():
            print(f"  {name}: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"No .bz2 archives in {RAW_DIR} (skipping bz2 extraction).")

    # Download + preprocess the raw datasets into binary LIBSVM files.
    for label, builder in (
        ("UCI Covertype", preprocess_covertype_uci),
        ("Gisette",       preprocess_gisette),
        ("real-sim",      preprocess_real_sim),
    ):
        print(f"\nBuilding {label} ...")
        try:
            path = builder()
            print(f"  {label}: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:  # keep going if one dataset's source is down
            print(f"  {label}: FAILED — {exc}")

    print("\nDone.")
