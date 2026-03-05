#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MACROS_DIR="$ROOT/aquaregia-macros"

# ── 读取主 crate 版本 ──────────────────────────────────────────────────────────
VERSION=$(grep '^version' "$ROOT/Cargo.toml" | head -1 | sed 's/.*"\(.*\)"/\1/')
echo ">>> 发布版本: $VERSION"

# ── 同步 macros 版本 ───────────────────────────────────────────────────────────
MACROS_VERSION=$(grep '^version' "$MACROS_DIR/Cargo.toml" | head -1 | sed 's/.*"\(.*\)"/\1/')
if [ "$MACROS_VERSION" != "$VERSION" ]; then
  echo ">>> 同步 aquaregia-macros 版本: $MACROS_VERSION → $VERSION"
  sed -i '' "s/^version = \"$MACROS_VERSION\"/version = \"$VERSION\"/" "$MACROS_DIR/Cargo.toml"
fi

# 同步主 Cargo.toml 里对 macros 的依赖版本
sed -i '' "s/aquaregia-macros = { path = \"aquaregia-macros\", version = \"[^\"]*\" }/aquaregia-macros = { path = \"aquaregia-macros\", version = \"$VERSION\" }/" "$ROOT/Cargo.toml"

# ── 发布 aquaregia-macros（若已存在则跳过）────────────────────────────────────
MACROS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "User-Agent: aquaregia-publish-script/1.0" \
  "https://crates.io/api/v1/crates/aquaregia-macros/$VERSION")

if [ "$MACROS_STATUS" = "200" ]; then
  echo ">>> aquaregia-macros $VERSION 已存在，跳过发布"
else
  echo ">>> 发布 aquaregia-macros ..."
  (cd "$MACROS_DIR" && cargo publish --registry crates-io)
  echo ">>> aquaregia-macros 已提交，等待 crates.io 索引同步..."
fi

# ── 轮询 crates.io，等待 macros 版本可见 ──────────────────────────────────────
POLL_INTERVAL=10
TIMEOUT=300
elapsed=0

while true; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "User-Agent: aquaregia-publish-script/1.0" \
    "https://crates.io/api/v1/crates/aquaregia-macros/$VERSION")

  if [ "$STATUS" = "200" ]; then
    echo ">>> aquaregia-macros $VERSION 已在 crates.io 上可见"
    break
  fi

  elapsed=$((elapsed + POLL_INTERVAL))
  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "错误：等待超时（${TIMEOUT}s），请手动检查 crates.io 后重试"
    exit 1
  fi

  echo "    等待中... (${elapsed}s / ${TIMEOUT}s)"
  sleep "$POLL_INTERVAL"
done

# ── 发布主 crate ───────────────────────────────────────────────────────────────
echo ">>> 发布 aquaregia ..."
(cd "$ROOT" && cargo publish --registry crates-io)
echo ">>> 发布完成！aquaregia $VERSION 已上线"
