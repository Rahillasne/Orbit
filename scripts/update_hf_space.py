#!/usr/bin/env python3
"""Update the HuggingFace Space (spaces/) for v1.2.0.

Run from the repo root:
    python scripts/update_hf_space.py

This script patches spaces/app.py to reflect v1.2.0 stats.
After running, review the diff and push to the HF Space repo.
"""

import re
from pathlib import Path

SPACES_DIR = Path("spaces")
APP_PY = SPACES_DIR / "app.py"


def patch_app():
    text = APP_PY.read_text()

    # 1. Update version badge: v1.1.0 -> v1.2.0
    text = text.replace(
        '<span class="orbit-badge badge-blue">v1.1.0</span>',
        '<span class="orbit-badge badge-blue">v1.2.0</span>',
    )

    # 2. Update rho badge: 0.85 -> 0.61 (LOOCV)
    text = text.replace(
        '<span class="orbit-badge badge-green">Spearman rho = 0.85</span>',
        '<span class="orbit-badge badge-green">LOOCV Spearman rho = 0.61</span>',
    )

    # 3. Update test count badge: 202 -> 399
    text = text.replace(
        '<span class="orbit-badge badge-purple">202 tests passed</span>',
        '<span class="orbit-badge badge-purple">399 tests passed</span>',
    )

    # 4. Update footer version
    text = text.replace("ORBIT v1.1.0", "ORBIT v1.2.0")

    APP_PY.write_text(text)
    print(f"Patched {APP_PY}")


def main():
    if not APP_PY.exists():
        print(f"ERROR: {APP_PY} not found. Run from the repo root.")
        return

    patch_app()

    print()
    print("Done! Next steps:")
    print("  1. Review changes:  git diff spaces/app.py")
    print("  2. Push to HF Space:  cd spaces && git push")
    print()
    print("Files that may also need updating in the HF Space:")
    print("  - spaces/data/benchmark_results.json (if you re-ran benchmarks)")
    print("  - spaces/data/demo_profile.json (if the profile format changed)")


if __name__ == "__main__":
    main()
