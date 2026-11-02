#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Check for new llama.cpp release
current=$(./scripts/llama-cpp-version.sh)
release=$(./scripts/llama-cpp-version.sh latest)
read -r latest latest_date <<<"$release"

echo "Current: $current"
echo "Latest:  $latest ($latest_date)"

if [ "$latest" = "$current" ]; then
  echo "Already up to date."
  exit 0
fi

./scripts/llama-cpp-version.sh set "$latest" "$latest_date"

echo "Updated $current -> $latest"

# Clean cached llama.cpp and rebuild
rm -rf build/_deps/llama_cpp-*
cmake -B build
cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

# Run tests
if ./build/tests; then
  echo "All tests passed."
else
  echo "Tests FAILED. Review changes before committing."
  exit 1
fi
