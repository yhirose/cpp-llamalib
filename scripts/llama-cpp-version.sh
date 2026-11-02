#!/usr/bin/env bash
# The single place that reads and writes the llama.cpp version this project
# pins: set(LLAMA_CPP_VERSION ...) in CMakeLists.txt and the
# "Tested against llama.cpp ..." line in README.md.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  echo "usage: $0 [get | latest | set TAG DATE]" >&2
  exit 2
}

# Print the pinned version.
get() {
  local version
  version=$(perl -ne 'if (/^set\(LLAMA_CPP_VERSION\s+(\S+)\)/) { print $1; exit }' CMakeLists.txt)
  if [ -z "$version" ]; then
    echo "set(LLAMA_CPP_VERSION ...) not found in CMakeLists.txt" >&2
    exit 1
  fi
  echo "$version"
}

# Print the latest upstream release as "TAG DATE".
latest() {
  gh api repos/ggml-org/llama.cpp/releases/latest --jq '.tag_name + " " + .published_at[:10]'
}

# Pin TAG (released on DATE) in CMakeLists.txt and README.md.
set_version() {
  local tag=$1 date=$2
  local readme="Tested against llama.cpp [$tag](https://github.com/ggml-org/llama.cpp/releases/tag/$tag), $date."

  # perl -i behaves the same on macOS and Linux, unlike sed -i.
  TAG=$tag perl -pi -e 's/^set\(LLAMA_CPP_VERSION\s+\S+\)/set(LLAMA_CPP_VERSION $ENV{TAG})/' CMakeLists.txt
  LINE=$readme perl -pi -e 's/^Tested against llama\.cpp .*/$ENV{LINE}/' README.md

  # A pattern that silently stops matching must fail here, not ship.
  if [ "$(get)" != "$tag" ]; then
    echo "Failed to pin $tag in CMakeLists.txt" >&2
    exit 1
  fi
  if ! grep -qxF "$readme" README.md; then
    echo "Failed to update the llama.cpp version in README.md" >&2
    exit 1
  fi
}

case "${1:-get}" in
  get)    get ;;
  latest) latest ;;
  set)    [ $# -eq 3 ] || usage; set_version "$2" "$3" ;;
  *)      usage ;;
esac
