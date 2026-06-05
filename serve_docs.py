"""
Build Sphinx docs and serve them on $PORT (default 5000).

Usage:
    python3 snn/serve_docs.py
    PORT=8000 python3 snn/serve_docs.py
"""

import http.server
import os
import socketserver
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_SRC = os.path.join(HERE, "docs")
DOCS_BUILD = os.path.join(DOCS_SRC, "_build", "html")
PORT = int(os.environ.get("PORT", 5000))


def install_deps():
    print("→ Installing Sphinx dependencies …")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "sphinx", "furo", "myst-parser", "sphinx-autodoc-typehints",
        ],
        check=True,
    )
    print("✓ Dependencies ready")


def build_docs():
    print("→ Building Sphinx docs …")
    result = subprocess.run(
        [
            sys.executable, "-m", "sphinx",
            "-b", "html",
            "-q",
            DOCS_SRC,
            DOCS_BUILD,
        ],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("⚠ Sphinx warnings/errors:")
        print(result.stdout[-3000:] if result.stdout else "")
        print(result.stderr[-3000:] if result.stderr else "")
        if result.returncode != 0 and not os.path.isdir(DOCS_BUILD):
            raise RuntimeError("Sphinx build failed and no output directory found.")
    else:
        print(f"✓ Docs built → {DOCS_BUILD}")


def serve():
    os.chdir(DOCS_BUILD)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # suppress per-request noise

    print(f"✓ Serving snn docs on http://0.0.0.0:{PORT}")
    print(f"  Open: http://localhost:{PORT}")

    with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    install_deps()
    build_docs()
    serve()
