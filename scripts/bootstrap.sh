#!/bin/sh
# Bootstrap the development environment from a clean checkout.
set -eu
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# XGBoost's macOS wheel links against Homebrew's OpenMP runtime.
if [ "$(uname)" = "Darwin" ] && ! [ -e /opt/homebrew/opt/libomp/lib/libomp.dylib ]; then
    brew install libomp
fi
uv python install 3.12
uv sync
uv run pre-commit install
