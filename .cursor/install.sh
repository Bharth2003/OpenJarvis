#!/usr/bin/env bash
# Cloud Agent install step for OpenJarvis.
#
# Mirrors the CI build in .github/workflows/ci.yml so the Cloud Agent
# environment matches what contributors and CI expect: a uv-managed Python
# virtualenv plus the native openjarvis_rust PyO3 extension.
#
# Safe to run repeatedly: every step converges to the same state.
set -euo pipefail

# 1. uv — Python package/venv manager. Install only if the base image lacks it.
#    The installer also adds ~/.local/bin to ~/.bashrc / ~/.profile so future
#    interactive shells find uv on PATH.
if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2. Python dependencies — same extras as CI's lint and test jobs.
uv sync --extra dev --extra framework-comparison --extra server

# 3. Native Rust extension (openjarvis_rust), built with maturin.
#    rust/rust-toolchain.toml pins Rust 1.88, so build from the rust/ directory
#    and let rustup auto-select (and install) that toolchain.
#    This MUST run after `uv sync`: uv prunes the editable openjarvis-rust
#    package (it lives in the desktop-native dependency group, not the synced
#    extras), and maturin reinstalls it here.
(
  cd rust
  uv run --project .. maturin develop \
    --manifest-path crates/openjarvis-python/Cargo.toml
)

echo "OpenJarvis environment ready."
