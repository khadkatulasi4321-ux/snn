"""
Build the snn Sphinx documentation and serve it on ``$PORT`` (default 5000).

Usage::

    # From anywhere after installing the package:
    python -m snn.docs.serve_docs

    # With a custom port:
    PORT=8080 python -m snn.docs.serve_docs

    # As a console script (if entry_points configured in setup.py):
    snn-docs

The script auto-installs Sphinx and its theme if they are not already present,
builds the HTML docs, then starts a simple HTTP server.
"""

from __future__ import annotations

import http.server
import os
import socketserver
import subprocess
import sys
from pathlib import Path


# ── Locate the Sphinx source directory ───────────────────────────────────────

def _find_docs_src() -> Path:
    """Return the path to the Sphinx source directory.

    Search order
    ------------
    1. ``DOCS_PATH`` environment variable (absolute or relative path).
    2. Next to the *repo root* when the package is installed in editable mode
       (``pip install -e .``).  The root is four levels above this file::

           <repo>/snn/snn/docs/serve_docs.py
                  ↑pkg  ↑snn ↑docs
           → <repo>/docs/

    3. A ``source/`` sub-directory bundled inside this ``snn.docs`` package
       (for regular PyPI installs where docs are shipped as package data).
    """
    env_path = os.environ.get("DOCS_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if p.is_dir():
            return p

    this_file = Path(__file__).resolve()

    # Editable install: this_file = <repo>/snn/snn/docs/serve_docs.py
    # repo root = this_file.parent × 3, then /docs
    candidate_editable = this_file.parent.parent.parent.parent / "docs"
    if candidate_editable.is_dir() and (candidate_editable / "conf.py").exists():
        return candidate_editable

    # Installed package with bundled docs
    candidate_bundled = this_file.parent / "source"
    if candidate_bundled.is_dir() and (candidate_bundled / "conf.py").exists():
        return candidate_bundled

    raise FileNotFoundError(
        "Could not locate the snn docs source directory.\n"
        "Options:\n"
        "  • Use an editable install:  pip install -e snn/\n"
        "  • Set DOCS_PATH=/path/to/docs\n"
        "  • Install with extras:      pip install snn[docs]"
    )


# ── Sphinx dependency bootstrap ───────────────────────────────────────────────

def _install_deps() -> None:
    pkgs = ["sphinx", "furo", "myst-parser", "sphinx-autodoc-typehints"]
    missing = []
    for pkg in pkgs:
        mod = pkg.split("[")[0].replace("-", "_")
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"→ Installing Sphinx dependencies: {', '.join(missing)} …")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + missing,
            check=True,
        )
        print("✓ Dependencies ready")


# ── Sphinx build ──────────────────────────────────────────────────────────────

def _build(docs_src: Path) -> Path:
    build_dir = docs_src / "_build" / "html"
    print("→ Building Sphinx docs …")
    result = subprocess.run(
        [
            sys.executable, "-m", "sphinx",
            "-b", "html",
            "-q",
            str(docs_src),
            str(build_dir),
        ],
        cwd=str(docs_src.parent),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr)[-4000:]
        print("⚠  Sphinx warnings / errors:\n" + tail)
        if not build_dir.is_dir():
            raise RuntimeError("Sphinx build failed and no output directory found.")
    else:
        print(f"✓ Docs built → {build_dir}")
    return build_dir


# ── HTTP server ───────────────────────────────────────────────────────────────

def _serve(build_dir: Path, port: int) -> None:
    os.chdir(build_dir)

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):  # suppress per-request noise
            pass

    print(f"✓ Serving snn docs on http://0.0.0.0:{port}")
    print(f"  Open:  http://localhost:{port}")
    print("  Press Ctrl-C to stop.\n")
    with socketserver.TCPServer(("", port), _QuietHandler) as httpd:
        httpd.serve_forever()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    port = int(os.environ.get("PORT", 5000))
    docs_src = _find_docs_src()
    print(f"  Docs source: {docs_src}")
    _install_deps()
    build_dir = _build(docs_src)
    _serve(build_dir, port)


if __name__ == "__main__":
    main()
