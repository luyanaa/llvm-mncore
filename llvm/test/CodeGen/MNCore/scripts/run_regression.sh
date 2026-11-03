#!/bin/bash
# Top-level MNCore regression: local FileCheck + remote assemble3 + emulator.
# Run from anywhere; paths are absolute.
#
#   run_regression.sh            # everything
#   run_regression.sh --local    # FileCheck only (no ssh)
set -u
TD=/Users/yanlu/Documents/llvm-mncore/llvm/test/CodeGen/MNCore
BD=/Users/yanlu/Documents/llvm-mncore/build-mncore
LLC=$BD/bin/llc
FC=$BD/bin/FileCheck

LOCAL_ONLY=0
[ "${1:-}" = "--local" ] && LOCAL_ONLY=1

# Tests that emit straight .vsm text (full llc run).
EMIT="peoperand-modes l1bml2bm-emit mv-emit mau-emit alu-emit mask-emit imm-fixed-emit"
# Tests checked via -run-pass=mncore-packetizer (MIR output).
PKT="mau-rules packetize-groups peoperand-conflict vfma-mwrite-shape mask-rules"
# Tests checked via -run-pass=mncore-hazard (MIR output).
HAZ="l2bmv-gap l1bset-hazard"
# Plain-IR tests (codegen-shape checks; the M1 register spelling is not
# yet assemble3-legal, so these are FileCheck only).
IRL="wide-imm"
# Tests that must be rejected (negative fixtures).
NEG="operand-placement"
# Tests whose vsm needs --instruction-mode flat on assemble3.
FLAT="peoperand-modes"

fail=0
echo "===== FileCheck ====="
for t in $EMIT; do
  $LLC -mtriple=mncore2 $TD/$t.mir -o - 2>/dev/null | $FC $TD/$t.mir >/dev/null 2>&1 \
    && echo "PASS $t" || { echo "FAIL $t"; fail=$((fail+1)); }
done
for t in $PKT; do
  $LLC -mtriple=mncore2 -run-pass=mncore-packetizer $TD/$t.mir -o - 2>/dev/null \
    | $FC $TD/$t.mir >/dev/null 2>&1 \
    && echo "PASS $t" || { echo "FAIL $t"; fail=$((fail+1)); }
done
for t in $HAZ; do
  $LLC -mtriple=mncore2 -run-pass=mncore-hazard $TD/$t.mir -o - 2>/dev/null \
    | $FC $TD/$t.mir >/dev/null 2>&1 \
    && echo "PASS $t" || { echo "FAIL $t"; fail=$((fail+1)); }
done
for t in $IRL; do
  $LLC -mtriple=mncore2 $TD/$t.ll -o - 2>/dev/null | $FC $TD/$t.ll >/dev/null 2>&1 \
    && echo "PASS $t" || { echo "FAIL $t"; fail=$((fail+1)); }
done
for t in $NEG; do
  out=$($LLC -mtriple=mncore2 $TD/$t.mir -o - 2>&1)
  if [ $? -eq 0 ] || ! printf '%s' "$out" | grep -q "illegal PE operand placement"; then
    echo "FAIL $t (expected a rejection)"; fail=$((fail+1))
  else
    echo "PASS $t"
  fi
done

if [ $LOCAL_ONLY = 0 ]; then
  echo "===== assemble3 + gpfn3_package_main ====="
  for t in $EMIT; do
    extra=""
    case " $FLAT " in *" $t "*) extra="--flat" ;; esac
    $TD/scripts/emulate_gpfn3.sh $extra $TD/$t.mir | grep -v '^emulate:'
  done
fi
echo "===== done (local failures: $fail) ====="
