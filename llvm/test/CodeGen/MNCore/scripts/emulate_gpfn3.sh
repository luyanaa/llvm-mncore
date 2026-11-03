#!/bin/bash
# Full regression pipeline for each MNCore MIR test:
#   llc (.mir -> .vsm)  ->  assemble3 (.vsm -> .asm)  ->  gpfn3_package_main (emulate)
# assemble3 / gpfn3_package_main run on 'suzhou' under Intel SDE -hsw.
#
# Usage:  emulate_gpfn3.sh [--flat] [--hazard] [--dump] <test.mir> [...]
#   --flat    assemble with --instruction-mode flat (needed by flat-address tests)
#   --hazard  enable hazard + deadlock checkers (default OFF: these tests reuse
#             addresses and are not authored to be data-race-free)
#   --dump    also run gpfn3 with `-d <dmp>` and copy the dump back to
#             /tmp/mncore_<test>.dmp for value checks with check_dump.py
#             (convention borrowed from taxift/openacc-mn-tuna-emu-demo)
#
# Requires build-mncore/bin/llc next to the repo root and ssh access to 'suzhou'.
set -u
# A missing ssh connection must never look like success: fail fast, and let
# the caller decide whether that is acceptable (run_regression.sh exits
# non-zero unless --local/--allow-offline was given).
if ! ssh -o BatchMode=yes -o ConnectTimeout=8 suzhou true 2>/dev/null; then
  echo "SKIP  (ssh suzhou unavailable)"
  exit 2
fi
LLC=/Users/yanlu/Documents/llvm-mncore/build-mncore/bin/llc
SDE=/opt/sde-external-10.13.1-2026-07-28-lin/sde64
ASM=/opt/pfn/pfcomp/codegen/build/tools/assemble3
EMU=/opt/pfn/pfcomp/codegen/build/tools/gpfn3_package_main

FLAT=0; HAZARD=0; DUMP=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --flat) FLAT=1 ;;
    --hazard) HAZARD=1 ;;
    --dump) DUMP=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

A3FLAGS=""
[ "$FLAT" = 1 ] && A3FLAGS="$A3FLAGS --instruction-mode flat"
[ "$HAZARD" = 0 ] && A3FLAGS="$A3FLAGS --disable-hazard-checker --disable-deadlock-checker"

pass=0; fail=0
for mir in "${ARGS[@]}"; do
  name=$(basename "$mir" .mir)
  vsm=$("$LLC" -mtriple=mncore2 "$mir" -o - 2>/dev/null \
        | grep -v '^\s*\.\|^\s*#\|^[A-Za-z_][A-Za-z0-9_]*:' | grep '\S')
  if [ -z "$vsm" ]; then
    echo "FAIL  $name  (llc produced no instructions)"
    fail=$((fail+1)); continue
  fi
  rv=/tmp/mncore_$name.vsm
  ra=/tmp/mncore_$name.asm
  printf '%s\n' "$vsm" | ssh suzhou "cat > $rv" 2>/dev/null || true
  if ! ssh suzhou "test -s $rv" 2>/dev/null; then
    echo "FAIL  $name  (could not stage $rv on suzhou)"
    fail=$((fail+1)); continue
  fi
  # Stage 2: assemble. Capture stderr; a non-empty diagnostic means failure.
  aout=$(ssh suzhou "$SDE -hsw -- $ASM $A3FLAGS --output-file $ra $rv 2>&1" | grep -v '^[ijm] [01]')
  if [ -n "$aout" ]; then
    echo "FAIL  $name  (assemble3)"
    echo "$aout" | sed 's/^/      /' | grep -iv '^\s*$' | head -8
    fail=$((fail+1)); continue
  fi
  # Stage 3: emulate. Log goes to a file; success = emulator exits 0.
  rd=/tmp/mncore_$name.dmp
  dflag=""
  [ $DUMP = 1 ] && dflag="-d $rd"
  eout=$(ssh suzhou "$SDE -hsw -- $EMU -i $ra -x /tmp/mncore_$name.log $dflag 2>&1")
  erc=$?
  if [ $erc -ne 0 ]; then
    echo "FAIL  $name  (gpfn3_package_main rc=$erc)"
    echo "$eout" | sed 's/^/      /' | head -8
    fail=$((fail+1)); continue
  fi
  if [ $DUMP = 1 ]; then
    ssh suzhou "cat $rd" > /tmp/mncore_$name.dmp 2>/dev/null
    echo "      dump: /tmp/mncore_$name.dmp"
  fi
  nsteps=$(printf '%s\n' "$vsm" | grep -c '\S')
  echo "PASS  $name  ($nsteps steps, emulated)"
  pass=$((pass+1))
done
echo "-----------------------------------------"
echo "emulate: $pass passed, $fail failed"
[ $fail -eq 0 ]
