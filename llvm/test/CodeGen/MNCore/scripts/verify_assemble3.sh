#!/bin/bash
# Round-trip .vsm output of each MNCore MIR test through PFN's assemble3
# on the 'suzhou' host (Intel SDE -hsw wrapper; tools are AVX2).
#
# Usage:  verify_assemble3.sh [--flat] [--no-hazard] <test.mir> [...]
#   --flat       assemble with --instruction-mode flat (for flat-address tests)
#   --no-hazard  disable hazard + deadlock checkers (syntax-only verification)
#
# Requires: build-mncore/bin/llc next to the repo root, ssh access to 'suzhou'.
set -u
LLC=/Users/yanlu/Documents/llvm-mncore/build-mncore/bin/llc
SDE=/opt/sde-external-10.13.1-2026-07-28-lin/sde64
ASM=/opt/pfn/pfcomp/codegen/build/tools/assemble3

FLAT=0; HAZARD=1
ARGS=()
for a in "$@"; do
  case "$a" in
    --flat) FLAT=1 ;;
    --no-hazard) HAZARD=0 ;;
    *) ARGS+=("$a") ;;
  esac
done

A3FLAGS=""
[ "$FLAT" = 1 ] && A3FLAGS="$A3FLAGS --instruction-mode flat"
[ "$HAZARD" = 0 ] && A3FLAGS="$A3FLAGS --disable-hazard-checker --disable-deadlock-checker"

pass=0; fail=0
for mir in "${ARGS[@]}"; do
  name=$(basename "$mir" .mir)
  # Strip ELF-asm boilerplate; keep only instruction lines.
  vsm=$("$LLC" -mtriple=mncore2 "$mir" -o - 2>/dev/null \
        | grep -v '^\s*\.\|^\s*#\|^[A-Za-z_][A-Za-z0-9_]*:' | grep '\S')
  remote=/tmp/mncore_verify_$name.vsm
  printf '%s\n' "$vsm" | ssh suzhou "cat > $remote" 2>/dev/null
  out=$(ssh suzhou "$SDE -hsw -- $ASM $A3FLAGS $remote 2>&1" | grep -v '^[ijm] [01]')
  if [ -z "$out" ]; then
    echo "PASS  $name"
    pass=$((pass+1))
  else
    echo "FAIL  $name"
    echo "$out" | sed 's/^/      /' | head -12
    fail=$((fail+1))
  fi
done
echo "-----------------------------------------"
echo "assemble3 verify: $pass passed, $fail failed"
