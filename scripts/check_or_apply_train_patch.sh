#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/third_party/gaussian-splatting"
PATCH="$ROOT/patches/gaussian_splatting_train_mcgs.diff"

mode="${1:-check}"

case "$mode" in
  check)
    if git -C "$TARGET" apply --check "$PATCH" 2>/dev/null; then
      echo "Patch can be applied cleanly."
    elif git -C "$TARGET" apply --reverse --check "$PATCH" 2>/dev/null; then
      echo "Patch is already applied."
    else
      echo "Patch cannot be applied cleanly and is not already applied." >&2
      exit 1
    fi
    ;;
  apply)
    if git -C "$TARGET" apply --reverse --check "$PATCH" 2>/dev/null; then
      echo "Patch is already applied."
    else
      git -C "$TARGET" apply "$PATCH"
      echo "Patch applied to $TARGET/train.py"
    fi
    ;;
  *)
    echo "Usage: $0 [check|apply]" >&2
    exit 2
    ;;
esac
