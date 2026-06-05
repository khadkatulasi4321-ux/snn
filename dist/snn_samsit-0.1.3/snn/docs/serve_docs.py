"""
Build and serve snn Sphinx documentation locally.

Usage:
    python -m snn.docs.serve_docs

Custom port:
    PORT=8080 python -m snn.docs.serve_docs

Custom docs path (override default discovery):
    DOCS_PATH=/path/to/docs python -m snn.docs.serve_docs
"""

from __future__ import annotations

import http.server
import os
import socketserver
import subprocess
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Resolve docs source
# ─────────────────────────────────────────────────────────────

def _find_docs_src() -> Path:
    """
    Resolve Sphinx docs source directory.

    Priority:
    1. DOCS_PATH environment variable
    2. Repo default: snn/snn/docs/docs
    3. Installed package fallback
    """

    # 1. Environment override
    env = os.environ.get("DOCS_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"DOCS_PATH set but invalid: {p}")

    this_file = Path(__file__).resolve()

    # ─────────────────────────────────────────────
    # 2. Default repo layout (your current structure)
    #
    # snn/snn/docs/serve_docs.py
    # → docs live at: snn/snn/docs/docs
    # ─────────────────────────────────────────────
    repo_docs = this_file.parents[1] / "docs" / "docs"

    if repo_docs.is_dir() and (repo_docs / "conf.py").exists():
        return repo_docs

    # ─────────────────────────────────────────────
    # 3. Editable install fallback (repo root/docs)
    # ─────────────────────────────────────────────
    editable_docs = this_file.parents[3] / "docs"
    if editable_docs.is_dir() and (editable_docs / "conf.py").exists():
        return editable_docs

    # ─────────────────────────────────────────────
    # 4. Package fallback (if bundled)
    # ─────────────────────────────────────────────
    pkg_docs = this_file.parent / "docs"
    if pkg_docs.is_dir() and (pkg_docs / "conf.py").exists():
        return pkg_docs

    raise FileNotFoundError(
        "Could not locate Sphinx docs.\n\n"
        "Expected one of:\n"
        "  • snn/snn/docs/docs (default repo layout)\n"
        "  • ../docs (editable install)\n"
        "  • DOCS_PATH=/custom/path\n"
    )


# ─────────────────────────────────────────────────────────────
# Install dependencies if missing
# ─────────────────────────────────────────────────────────────

def _install_deps() -> None:
    pkgs = [
        "sphinx",
        "furo",
        "myst-parser",
        "sphinx-autodoc-typehints",
    ]

    missing = []
    for pkg in pkgs:
        mod = pkg.replace("-", "_").split("[")[0]
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"→ Installing docs dependencies: {', '.join(missing)}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *missing],
            check=True,
        )
        print("✓ Dependencies installed")


# ─────────────────────────────────────────────────────────────
# Build docs
# ─────────────────────────────────────────────────────────────

def _build(docs_src: Path) -> Path:
    build_dir = docs_src / "_build" / "html"

    print("→ Building Sphinx docs...")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "-q",
            str(docs_src),
            str(build_dir),
        ],
        cwd=str(docs_src.parent),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("⚠ Sphinx warnings/errors:\n")
        print((result.stdout + result.stderr)[-4000:])
        if not build_dir.exists():
            raise RuntimeError("Sphinx build failed")

    print(f"✓ Docs built at: {build_dir}")
    return build_dir


# ─────────────────────────────────────────────────────────────
# Serve docs
# ─────────────────────────────────────────────────────────────

def _serve(build_dir: Path, port: int) -> None:
    os.chdir(build_dir)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # silence logs
            pass

    print(f"\n✓ Serving docs at http://localhost:{port}")
    print("  Press Ctrl+C to stop\n")

    with socketserver.TCPServer(("", port), Handler) as httpd:
        httpd.serve_forever()


# ─────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────

def main() -> None:
    port = int(os.environ.get("PORT", "5000"))

    docs_src = _find_docs_src()
    print(f"📚 Docs source: {docs_src}")

    _install_deps()
    build_dir = _build(docs_src)
    _serve(build_dir, port)


if __name__ == "__main__":
    main()
