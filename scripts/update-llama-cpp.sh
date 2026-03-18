#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Check for new llama.cpp release
current=$(sed -n 's/.*GIT_TAG\s\{1,\}\(\S\{1,\}\).*/\1/p' CMakeLists.txt)
latest=$(gh api repos/ggml-org/llama.cpp/releases/latest --jq '.tag_name')
latest_date=$(gh api repos/ggml-org/llama.cpp/releases/latest --jq '.published_at[:10]')

echo "Current: $current"
echo "Latest:  $latest ($latest_date)"

if [ "$latest" = "$current" ]; then
  echo "Already up to date."
  exit 0
fi

# Update CMakeLists.txt
sed -i '' "s/GIT_TAG\s\{1,\}\S\{1,\}/GIT_TAG        $latest/" CMakeLists.txt

# Update README.md
sed -i '' "s|llama.cpp \[.*\](https://github.com/ggml-org/llama.cpp/releases/tag/[^)]*), [0-9-]*\.|llama.cpp [$latest](https://github.com/ggml-org/llama.cpp/releases/tag/$latest), $latest_date.|" README.md

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
