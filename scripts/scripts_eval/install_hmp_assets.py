"""Install user-downloaded official HMP archives without handling credentials."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HMP_ROOT = ROOT / "thirdparty" / "HMP"
DOWNLOADS = HMP_ROOT / "downloads"


def extract(name: str, destination: Path, strip_top_level: str | None = None) -> None:
    archive = DOWNLOADS / name
    if not archive.is_file():
        raise FileNotFoundError(f"Missing official archive: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive} -> {destination}")
    with zipfile.ZipFile(archive) as handle:
        if strip_top_level is None:
            handle.extractall(destination)
            return
        prefix = strip_top_level.rstrip("/") + "/"
        for member in handle.infolist():
            if not member.filename.startswith(prefix):
                continue
            relative = member.filename[len(prefix):]
            if not relative:
                continue
            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def main() -> None:
    # The manually downloaded archives include their destination directory as
    # a wrapper.  Strip it here so HMP sees data/body_models/mano rather than
    # data/body_models/body_models/mano (and likewise for PyMAF-X/data).
    extract("body_models.zip", HMP_ROOT / "data" / "body_models", "body_models")
    extract("data.zip", HMP_ROOT / "external" / "PyMAF-X" / "data", "data")
    result_root = HMP_ROOT / "outputs" / "generative" / "results"
    extract("model.zip", result_root)
    normalization_root = HMP_ROOT / "data" / "amass" / "generative"
    normalization_root.mkdir(parents=True, exist_ok=True)
    for filename in ("mean-neutral-128-30fps.pt", "std-neutral-128-30fps.pt"):
        source = result_root / "model" / filename
        if source.exists():
            shutil.copy2(source, normalization_root / filename)
    print("[done] Official HMP assets installed")


if __name__ == "__main__":
    main()
