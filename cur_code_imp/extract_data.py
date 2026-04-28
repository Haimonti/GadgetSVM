import bz2
import shutil
from pathlib import Path

from config import RAW_DIR, DATA_DIR


def extract_bz2(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.stem  # strips .bz2
    with bz2.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dest


def extract_all() -> dict[str, Path]:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw folder not found: {RAW_DIR}")
    archives = sorted(RAW_DIR.glob("*.bz2"))
    if not archives:
        raise FileNotFoundError(f"No .bz2 files found in {RAW_DIR}")
    paths = {}
    for src in archives:
        out = extract_bz2(src, DATA_DIR)
        paths[src.stem] = out
        print(f"[{src.name}] extracted → {out}")
    return paths


if __name__ == "__main__":
    extracted = extract_all()
    print("\nDone.")
    for name, path in extracted.items():
        size_mb = path.stat().st_size / 1e6
        print(f"  {name}: {path}  ({size_mb:.1f} MB)")
