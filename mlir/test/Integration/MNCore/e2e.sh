#!/usr/bin/env bash
# End-to-end: mncore dialect -> LLVM IR -> llc -> .vsm text.
#
# Usage: BUILD_DIR=/path/to/build-mlir ./e2e.sh
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
build="${BUILD_DIR:-$here/../../../..//build-mlir}"

mlir_opt="$build/bin/mlir-opt"
mlir_translate="$build/bin/mlir-translate"
llc="$build/bin/llc"

for tool in "$mlir_opt" "$mlir_translate" "$llc"; do
  if [[ ! -x "$tool" ]]; then
    echo "missing tool: $tool (set BUILD_DIR)" >&2
    exit 1
  fi
done

"$mlir_opt" "$here/fma.mlir" --convert-mncore-to-llvm \
  | "$mlir_translate" --mlir-to-llvmir \
  | "$llc" -mtriple=mncore2 -filetype=asm -o -
