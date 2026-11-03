#!/bin/bash
# Round-trip .vsm output of each MNCore test through PFN's assemble3 on the
# 'suzhou' host (Intel SDE -hsw wrapper).
#
# Usage:  verify_assemble3.sh [--flat] [--no-hazard] <test> [...]
#   --flat       assemble with --instruction-mode flat
#   --no-hazard  disable the hazard + deadlock checkers (syntax only)
#
# Multi-function inputs are split per function before assembly: the hazard
# checker must see each kernel as an independent instruction stream, otherwise
# a slot reused by one function looks like a cross-function data race.
#
# Exit status is non-zero if any stage fails. llc, ssh and assemble3 are
# checked independently; the remote file is unique per run so a stale
# /tmp artifact can never turn into a false PASS.
set -u -o pipefail
LLC=${LLC:-/Users/yanlu/Documents/llvm-mncore/build-mncore/bin/llc}
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

if ! ssh -o BatchMode=yes -o ConnectTimeout=8 suzhou true 2>/dev/null; then
  echo "SKIP  (ssh suzhou unavailable)"
  exit 2
fi

pass=0; fail=0
for src in "${ARGS[@]}"; do
  name=$(basename "$src")
  name=${name%.*}
  if ! full=$("$LLC" -mtriple=mncore2 "$src" -o - 2>/dev/null \
               | grep -v '^\s*\.' | grep -v '^\s*#'); then
    echo "FAIL  $name  (llc failed)"; fail=$((fail+1)); continue
  fi
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/mncore_fn.XXXXXX")
  printf '%s\n' "$full" | awk -v dir="$tmp" '
    /^[A-Za-z_][A-Za-z0-9_]*:/ { n=$1; sub(/:.*/, "", n); file=dir"/"n".vsm"; next }
    file != "" && NF { print >> file }
  '
  shopt -s nullglob
  files=("$tmp"/*.vsm)
  shopt -u nullglob
  if [ ${#files[@]} -eq 0 ]; then
    echo "FAIL  $name  (llc produced no instructions)"
    rm -rf "$tmp"; fail=$((fail+1)); continue
  fi

  bfail=0
  for f in "${files[@]}"; do
    fn=$(basename "$f" .vsm)
    remote=$(ssh suzhou "mktemp /tmp/mncore_verify_XXXXXX.vsm" 2>/dev/null) || {
      echo "FAIL  $name/$fn  (ssh mktemp failed)"
      bfail=1; break; }
    remote_asm=${remote%.vsm}.asm
    if ! ssh suzhou "cat > '$remote'" < "$f"; then
      echo "FAIL  $name/$fn  (ssh staging failed)"
      bfail=1; break
    fi
    if ! ssh suzhou "test -s '$remote'"; then
      echo "FAIL  $name/$fn  (staged file empty)"
      bfail=1; break
    fi
    aout=$(ssh suzhou "$SDE -hsw -- $ASM $A3FLAGS --output-file '$remote_asm' '$remote' 2>&1")
    arc=$?
    diag=$(printf '%s\n' "$aout" | grep -v '^[ijm] [01]' | grep -v '^\s*$')
    if [ $arc -ne 0 ] || [ -n "$diag" ]; then
      echo "FAIL  $name/$fn  (assemble3 rc=$arc)"
      printf '%s\n' "$diag" | sed 's/^/      /' | head -8
      bfail=1
      ssh suzhou "rm -f '$remote' '$remote_asm'" 2>/dev/null || true
      break
    fi
    ssh suzhou "rm -f '$remote' '$remote_asm'" 2>/dev/null || true
  done
  rm -rf "$tmp"
  if [ $bfail -eq 0 ]; then
    echo "PASS  $name  (${#files[@]} function(s))"
    pass=$((pass+1))
  else
    fail=$((fail+1))
  fi
done
echo "-----------------------------------------"
echo "assemble3 verify: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
