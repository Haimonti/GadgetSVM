import bz2
import os
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

BZ2_FILES = {
    "train": Path(__file__).parent / "rcv1_train.binary.bz2",
    "test": Path(__file__).parent / "rcv1_test.binary.bz2",
}


def extract_bz2(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.stem  # strips .bz2, keeps rcv1_train.binary
    with bz2.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dest


def extract_all() -> dict[str, Path]:
    paths = {}
    for split, src in BZ2_FILES.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing archive: {src}")
        dest_dir = TRAIN_DIR if split == "train" else TEST_DIR
        out = extract_bz2(src, dest_dir)
        paths[split] = out
        print(f"[{split}] extracted → {out}")
    return paths


if __name__ == "__main__":
    extracted = extract_all()
    print("\nDone.")
    for split, path in extracted.items():
        size_mb = path.stat().st_size / 1e6
        print(f"  {split}: {path}  ({size_mb:.1f} MB)")
