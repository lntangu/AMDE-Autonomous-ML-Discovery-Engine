"""Register AMDE's src directory in this virtual environment without network install."""
from __future__ import annotations

import argparse
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "src").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-packages",
        default=sysconfig.get_paths().get("purelib"),
        help="site-packages directory (defaults to the active Python environment)",
    )
    args = parser.parse_args()
    target_dir = Path(args.site_packages).resolve() if args.site_packages else None
    if not SRC.is_dir():
        parser.error(f"source tree not found: {SRC}")
    if target_dir is None or not target_dir.is_dir():
        parser.error(f"site-packages directory not found: {target_dir}")
    pth_path = target_dir / "amde_src.pth"
    pth_path.write_text(str(SRC) + "\n", encoding="utf-8")
    print(f"Registered local source path: {SRC}")
    print(f"Wrote: {pth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
