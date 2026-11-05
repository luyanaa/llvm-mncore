#!/usr/bin/env bash
# End-to-end: mncore dialect -> LLVM IR -> llc -> .vsm text.
#
# Usage: BUILD_DIR=/path/to/build-mlir ./e2e.sh [input.mlir]
#
# With no argument, every *.mlir in this directory is lowered. A file whose
# pipeline fails, or that lowers to zero instructions, is a failure: a kernel
# that silently disappears is worse than one that errors.
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

run_one() {
  local input="$1"
  "$mlir_opt" "$input" --convert-mncore-to-llvm \
    | "$mlir_translate" --mlir-to-llvmir \
    | "$llc" -mtriple=mncore2 -filetype=asm -o -
}

# Single-file mode keeps the old behaviour (input on stdout, no header).
if [[ $# -ge 1 ]]; then
  run_one "$1"
  exit 0
fi

total=0
fail=0
for input in "$here"/*.mlir; do
  name="$(basename "$input" .mlir)"
  total=$((total + 1))
  echo "===== $name ====="
  if ! out="$(run_one "$input" 2>&1)"; then
    printf '%s\n' "$out" | tail -5
    echo "FAIL $name (pipeline failed)"
    fail=$((fail + 1))
    continue
  fi
  printf '%s\n' "$out"
  if ! printf '%s\n' "$out" | grep -qE '^[[:space:]]+[a-z]'; then
    echo "FAIL $name (no instructions emitted)"
    fail=$((fail + 1))
  fi
done
echo "-----------------------------------------"
echo "e2e: $((total - fail)) passed, $fail failed"
[[ $fail -eq 0 ]]
