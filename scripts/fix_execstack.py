"""Clear PT_GNU_STACK.X on every .so under .venv/lib/.../{jaxlib,nvidia}/.

The CUDA jaxlib wheel ships shared objects whose PT_GNU_STACK program header
has the executable flag set (`flags=0x7` = R+W+X). Some kernels — including
the AIchor / Linux 6.8 + LSM combo — refuse to load shared objects with an
executable stack and raise:

    ImportError: cannot enable executable stack as shared object requires:
                 Invalid argument

This script scans every `.so` under the venv's jaxlib + nvidia directories,
finds the `PT_GNU_STACK` program header in each ELF64 file, and clears the
PF_X bit. Idempotent: re-running on already-patched files is a no-op.

Usage:
    python scripts/fix_execstack.py
    python scripts/fix_execstack.py --venv /custom/path/.venv
"""

import argparse
import glob
import os
import struct
import sys

PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def patch_so(path: str) -> str:
    """Return one of: 'fixed', 'already_clean', 'not_elf', 'no_gnu_stack'."""
    try:
        with open(path, "r+b") as f:
            hdr = f.read(64)
            if len(hdr) < 64 or hdr[:4] != b"\x7fELF" or hdr[4] != 2:
                return "not_elf"
            e_phoff = struct.unpack("<Q", hdr[0x20:0x28])[0]
            e_phentsize = struct.unpack("<H", hdr[0x36:0x38])[0]
            e_phnum = struct.unpack("<H", hdr[0x38:0x3A])[0]

            for i in range(e_phnum):
                off = e_phoff + i * e_phentsize
                f.seek(off)
                ph = f.read(8)
                p_type, p_flags = struct.unpack("<II", ph)
                if p_type != PT_GNU_STACK:
                    continue
                if not (p_flags & PF_X):
                    return "already_clean"
                f.seek(off + 4)
                f.write(struct.pack("<I", p_flags & ~PF_X))
                return "fixed"
            return "no_gnu_stack"
    except OSError as e:
        print(f"  SKIP {path}: {e}", file=sys.stderr)
        return "not_elf"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--venv",
        default=".venv",
        help="path to the venv (default: ./.venv)",
    )
    args = ap.parse_args()

    site_pkgs_glob = os.path.join(args.venv, "lib", "python*", "site-packages")
    site_pkgs_dirs = glob.glob(site_pkgs_glob)
    if not site_pkgs_dirs:
        print(f"ERROR: no site-packages under {site_pkgs_glob}", file=sys.stderr)
        return 1

    targets = []
    for sp in site_pkgs_dirs:
        for sub in ("jaxlib", "nvidia"):
            root = os.path.join(sp, sub)
            if os.path.isdir(root):
                for path in glob.glob(os.path.join(root, "**", "*.so*"), recursive=True):
                    if not os.path.islink(path) and os.path.isfile(path):
                        targets.append(path)

    if not targets:
        print(f"WARNING: no .so files under {site_pkgs_dirs}/{{jaxlib,nvidia}}")
        return 0

    counts = {"fixed": 0, "already_clean": 0, "no_gnu_stack": 0, "not_elf": 0}
    for path in targets:
        result = patch_so(path)
        counts[result] += 1
        if result == "fixed":
            print(f"  FIXED  {os.path.relpath(path, args.venv)}")

    print(
        f"\nscanned={len(targets)}  "
        f"fixed={counts['fixed']}  "
        f"already_clean={counts['already_clean']}  "
        f"no_gnu_stack={counts['no_gnu_stack']}  "
        f"non_elf={counts['not_elf']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
