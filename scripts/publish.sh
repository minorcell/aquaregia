#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
echo ">>> Publishing aquaregia v$VERSION"

# ── Preflight ────────────────────────────────────────────────────────────────
echo ">>> Preflight..."

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?$ ]]; then
    echo "ERROR: invalid semver in Cargo.toml: '$VERSION'."
    exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo "ERROR: current branch is '$BRANCH'; must publish from main."
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is dirty; commit, stash, or .gitignore first."
    exit 1
fi

git fetch origin main --quiet
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
    echo "ERROR: local HEAD differs from origin/main; sync first."
    exit 1
fi

if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    echo "ERROR: tag v$VERSION already exists locally."
    exit 1
fi
if git ls-remote --tags origin "refs/tags/v$VERSION" | grep -q "refs/tags/v$VERSION$"; then
    echo "ERROR: tag v$VERSION already exists on origin."
    exit 1
fi

if curl -fsSL -A "publish-script/1.0" "https://crates.io/api/v1/crates/aquaregia/$VERSION" 2>/dev/null | grep -q '"crate_size"'; then
    echo "ERROR: aquaregia $VERSION is already published on crates.io; bump the version."
    exit 1
fi

echo ">>> Running tests..."
cargo test --quiet

echo ">>> cargo publish --dry-run..."
cargo publish --dry-run --registry crates-io

# Re-check Cargo.toml hasn't changed during preflight, so the tag we will push
# matches exactly what cargo publish is about to upload.
CURRENT_VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
if [ "$CURRENT_VERSION" != "$VERSION" ]; then
    echo "ERROR: Cargo.toml version changed during run (was '$VERSION', now '$CURRENT_VERSION')."
    exit 1
fi

# ── Publish (irreversible) ───────────────────────────────────────────────────
echo ">>> Publishing to crates.io..."
cargo publish --registry crates-io

# ── Tag and push ─────────────────────────────────────────────────────────────
echo ">>> Tagging v$VERSION and pushing..."
git tag -a "v$VERSION" -m "v$VERSION"
git push origin "v$VERSION"

echo ">>> Done: aquaregia v$VERSION published, tag pushed."
