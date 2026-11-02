#!/usr/bin/env bash
# Tag HEAD with the llama.cpp version it is tested against (e.g. v0.8.0),
# unless the remote already has that tag. Library-only releases (v0.8.0.1)
# are tagged by hand.
set -euo pipefail
cd "$(dirname "$0")/.."

remote=${1:-origin}
version=$(./scripts/llama-cpp-version.sh)

if git ls-remote --exit-code --tags "$remote" "refs/tags/$version" >/dev/null; then
  echo "Tag $version already exists."
  exit 0
fi

git tag -a "$version" -m "Tested against llama.cpp $version"
git push "$remote" "refs/tags/$version"
echo "Tagged $(git rev-parse --short HEAD) as $version."
