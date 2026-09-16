#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TARGET="${1:-main_v2}"
case "$TARGET" in main_v2|draft_v1) ;; *) echo "Usage: $0 [main_v2|draft_v1]" >&2; exit 2;; esac
pdflatex -interaction=nonstopmode -halt-on-error "$TARGET.tex"
# Some TeX installations provide a wrapper; the original binary is equivalent.
if command -v bibtex.original >/dev/null 2>&1; then
  bibtex.original "$TARGET"
else
  bibtex "$TARGET"
fi
pdflatex -interaction=nonstopmode -halt-on-error "$TARGET.tex"
pdflatex -interaction=nonstopmode -halt-on-error "$TARGET.tex"
