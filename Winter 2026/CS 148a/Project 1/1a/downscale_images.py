#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image, ImageOps


def _iter_images(indir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in indir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downscale images to a fixed square size.")
    parser.add_argument("--in", dest="indir", default="all", help="Input directory (default: all)")
    parser.add_argument("--out", dest="outdir", default="all_1024", help="Output directory (default: all_1024)")
    parser.add_argument("--size", type=int, default=1024, help="Output width/height in pixels (default: 1024)")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite originals in the input directory (ignores --out).",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    indir = Path(args.indir)
    if not indir.exists():
        # Common on macOS: case-insensitive FS might store as ALL while user types all.
        alt = Path(str(indir).upper())
        if alt.exists():
            indir = alt
        else:
            raise SystemExit(f"Input directory not found: {args.indir!r}")

    outdir = indir if args.inplace else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    images = _iter_images(indir)
    if not images:
        print(f"No images found in {indir}")
        return 0

    for src in images:
        dst = outdir / src.name
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im = im.resize((args.size, args.size), resample=Image.Resampling.LANCZOS)

            tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
            save_kwargs: dict = {}
            if src.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs.update({"quality": int(args.quality), "optimize": True, "progressive": True})
            im.save(tmp, **save_kwargs)
            os.replace(tmp, dst)

    print(f"Done: wrote {len(images)} image(s) to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
