import sys
import bz2
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DIR, DATA_DIR



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


if __name__ == "__main__":
    extracted = extract_all()
    print("\nDone.")
    for name, path in extracted.items():
        size_mb = path.stat().st_size / 1e6
        print(f"  {name}: {path}  ({size_mb:.1f} MB)")
