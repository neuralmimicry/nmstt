#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="$(uname -m)"

if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "arm64" ]]; then
  echo "This installer targets arm64/aarch64 hosts. Detected: ${ARCH}" >&2
  exit 1
fi

INSTALL_DIR="${INSTALL_DIR:-/opt/nmstt}"
SERVICE_NAME="${SERVICE_NAME:-nmstt.service}"
ENV_PATH="${ENV_PATH:-/etc/default/nmstt}"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}"
BUILD_RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"

echo "[1/5] Building release binary for native CPU..."
cd "${SCRIPT_DIR}"
RUSTFLAGS="${BUILD_RUSTFLAGS}" cargo build --release

echo "[2/5] Installing binary and systemd unit..."
sudo install -d -m 755 "${INSTALL_DIR}"
sudo install -m 755 "${SCRIPT_DIR}/target/release/nmstt" "${INSTALL_DIR}/nmstt"
sudo install -m 644 "${SCRIPT_DIR}/nmstt.service" "${UNIT_PATH}"

if [[ ! -f "${ENV_PATH}" ]]; then
  echo "[3/5] Installing default env preset to ${ENV_PATH}..."
  sudo install -m 644 "${SCRIPT_DIR}/native_arm64_46core.env" "${ENV_PATH}"
else
  echo "[3/5] Existing ${ENV_PATH} found; leaving it unchanged."
fi

echo "[4/5] Reloading and enabling systemd service..."
sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"

echo "[5/5] Restarting and showing status..."
sudo systemctl restart "${SERVICE_NAME}"
sudo systemctl --no-pager --full status "${SERVICE_NAME}"

echo "Completed native arm64 deployment for ${SERVICE_NAME}."
