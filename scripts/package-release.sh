#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: package-release.sh [options]

Build and package nmstt release artifacts.

Options:
  --version VERSION      Expected Cargo package version.
  --deb-version VER      Debian package version. Default: Cargo version.
  --deb-channel-alias    Optional stable Debian channel alias filename prefix.
                         Example: latest-main -> nmstt_latest-main_amd64.deb
  --input-dir DIR        Directory containing raw release binaries.
                         Default: ./raw-binaries
  --output-dir DIR       Directory to receive packaged artifacts.
  -h, --help             Show this help text.

Examples:
  ./scripts/package-release.sh --version 0.1.0 --output-dir ./dist
  ./scripts/package-release.sh --version 0.1.0 --deb-version 0.1.0~dev.12.abcd123 --input-dir ./raw-binaries --output-dir ./dist
USAGE
}

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

sha256_tool() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf 'sha256sum\n'
  elif command -v shasum >/dev/null 2>&1; then
    printf 'shasum -a 256\n'
  else
    die "sha256sum or shasum is required"
  fi
}

read_package_version() {
  awk '
    BEGIN { in_package=0 }
    /^\[package\]$/ { in_package=1; next }
    /^\[/ { in_package=0 }
    in_package && /^version[[:space:]]*=/ {
      gsub(/"/, "", $3)
      print $3
      exit
    }
  ' Cargo.toml
}

host_to_deb_arch() {
  case "$1" in
    x86_64-unknown-linux-gnu) printf 'amd64\n' ;;
    aarch64-unknown-linux-gnu) printf 'arm64\n' ;;
    *) return 1 ;;
  esac
}

create_deb() {
  local binary_path="$1"
  local deb_arch="$2"
  local deb_path="$3"
  local stage_root="$4"

  local deb_root="${stage_root}/${deb_arch}"
  rm -rf "$deb_root"
  mkdir -p "$deb_root/DEBIAN" "$deb_root/usr/local/bin" "$deb_root/opt/nmstt/models"

  install -m 0755 "$binary_path" "$deb_root/usr/local/bin/nmstt"
  if [[ -d "${REPO_ROOT}/models" ]]; then
    cp -a "${REPO_ROOT}/models/." "$deb_root/opt/nmstt/models/"
  fi

  cat >"$deb_root/DEBIAN/control" <<EOF
Package: nmstt
Version: ${DEB_VERSION}
Section: utils
Priority: optional
Architecture: ${deb_arch}
Maintainer: NeuralMimicry
Depends: ca-certificates, libc6
Description: NeuralMimicry Speech-to-Text service binary and model assets
EOF

  dpkg-deb --build --root-owner-group "$deb_root" "$deb_path" >/dev/null
}

VERSION=
DEB_VERSION=
DEB_CHANNEL_ALIAS=
INPUT_DIR=./raw-binaries
OUTPUT_DIR=

while (($#)); do
  case "$1" in
    --version)
      shift
      (($#)) || die "--version requires a value"
      VERSION="$1"
      ;;
    --deb-version)
      shift
      (($#)) || die "--deb-version requires a value"
      DEB_VERSION="$1"
      ;;
    --deb-channel-alias)
      shift
      (($#)) || die "--deb-channel-alias requires a value"
      DEB_CHANNEL_ALIAS="$1"
      ;;
    --input-dir)
      shift
      (($#)) || die "--input-dir requires a value"
      INPUT_DIR="$1"
      ;;
    --output-dir)
      shift
      (($#)) || die "--output-dir requires a value"
      OUTPUT_DIR="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
  shift
done

[[ -n "$OUTPUT_DIR" ]] || die "--output-dir is required"
[[ -d "$INPUT_DIR" ]] || die "input directory not found: $INPUT_DIR"
command -v dpkg-deb >/dev/null 2>&1 || die "dpkg-deb is required"

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PACKAGE_VERSION=$(read_package_version)
[[ -n "$PACKAGE_VERSION" ]] || die "Unable to determine Cargo package.version"
if [[ -n "$VERSION" && "$VERSION" != "$PACKAGE_VERSION" ]]; then
  die "Cargo.toml version ${PACKAGE_VERSION} does not match requested version ${VERSION}"
fi
if [[ -z "$DEB_VERSION" ]]; then
  DEB_VERSION="$PACKAGE_VERSION"
fi

INPUT_DIR=$(cd "$INPUT_DIR" && pwd)
OUTPUT_DIR=$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)
find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

mapfile -t binaries < <(find "$INPUT_DIR" -maxdepth 1 -type f -name 'nmstt-*' | sort)
[[ ${#binaries[@]} -gt 0 ]] || die "no raw nmstt binaries found in ${INPUT_DIR}"

stage_root="$OUTPUT_DIR/.deb-stage"
mkdir -p "$stage_root"

declare -A seen_arch
for binary_path in "${binaries[@]}"; do
  binary_name="$(basename "$binary_path")"
  if [[ "$binary_name" =~ ^nmstt-.+-(x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu)$ ]]; then
    host="${BASH_REMATCH[1]}"
  else
    log "Skipping unrecognized raw binary name: ${binary_name}"
    continue
  fi

  deb_arch="$(host_to_deb_arch "$host" || true)"
  [[ -n "$deb_arch" ]] || die "unsupported binary host target: ${host}"

  tar -C "$INPUT_DIR" -czf "$OUTPUT_DIR/${binary_name}.tar.gz" "$binary_name"

  if [[ -z "${seen_arch[$deb_arch]:-}" ]]; then
    deb_output="$OUTPUT_DIR/nmstt_${DEB_VERSION}_${deb_arch}.deb"
    create_deb "$binary_path" "$deb_arch" "$deb_output" "$stage_root"
    if [[ -n "$DEB_CHANNEL_ALIAS" ]]; then
      cp "$deb_output" "$OUTPUT_DIR/nmstt_${DEB_CHANNEL_ALIAS}_${deb_arch}.deb"
    fi
    seen_arch["$deb_arch"]=1
  fi
done

rm -rf "$stage_root"

CHECKSUM_PATH="$OUTPUT_DIR/nmstt-${DEB_VERSION}.sha256.txt"
checksum_cmd=$(sha256_tool)
(
  cd "$OUTPUT_DIR"
  artifacts=()
  for artifact in *; do
    [[ -f "$artifact" ]] || continue
    artifacts+=("$artifact")
  done
  [[ ${#artifacts[@]} -gt 0 ]] || die "no packaged artifacts were produced"
  $checksum_cmd "${artifacts[@]}" >"$(basename "$CHECKSUM_PATH")"
)

log
log "packaged nmstt release artifacts:"
find "$OUTPUT_DIR" -maxdepth 1 -type f | sort | sed 's#^#  #'
