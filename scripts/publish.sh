#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── 读取主 crate 版本 ──────────────────────────────────────────────────────────
VERSION=$(grep '^version' "$ROOT/Cargo.toml" | head -1 | sed 's/.*"\(.*\)"/\1/')
echo ">>> 发布版本: $VERSION"

# ── 发布主 crate ───────────────────────────────────────────────────────────────
echo ">>> 发布 aquaregia ..."
(cd "$ROOT" && cargo publish --registry crates-io)
echo ">>> 发布完成！aquaregia $VERSION 已上线"
