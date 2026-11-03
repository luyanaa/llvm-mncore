#!/bin/bash
# Top-level MNCore regression: local FileCheck + remote assemble3 + emulator.
# Run from anywhere; paths are absolute.
#
#   run_regression.sh                 # everything; non-zero on ANY failure
#   run_regression.sh --local         # FileCheck only
#   run_regression.sh --allow-offline # FileCheck + remote if reachable;
#                                      # an unreachable ssh is not a failure
#   run_regression.sh --strict        # add assemble3 --hazard on remote runs
set -u
TD=/Users/yanlu/Documents/llvm-mncore/llvm/test/CodeGen/MNCore
BD=/Users/yanlu/Documents/llvm-mncore/build-mncore
LLC=$BD/bin/llc
FC=$BD/bin/FileCheck

LOCAL_ONLY=0; ALLOW_OFFLINE=0; STRICT=0
for a in "$@"; do
  case "$a" in
    --local) LOCAL_ONLY=1; ALLOW_OFFLINE=1 ;;
    --allow-offline) ALLOW_OFFLINE=1 ;;
    --strict) STRICT=1 ;;
  esac
done

# Tests that emit straight .vsm text (full llc run).
EMIT="peoperand-modes l1bml2bm-emit mv-emit mau-emit alu-emit mask-emit imm-fixed-emit lmbar golden golden_mem"
# Tests checked via -run-pass=mncore-packetizer (MIR output).
PKT="mau-rules packetize-groups peoperand-conflict vfma-mwrite-shape mask-rules"
# Tests checked via -run-pass=mncore-hazard (MIR output).
HAZ="l2bmv-gap l1bset-hazard"
# Hazard tests that read stderr (attribution report).
HAZR="hazard-report region-overlap"
# Plain-IR tests (codegen-shape checks; the M1 register spelling is not
# yet assemble3-legal, so these are FileCheck only).
IRL="wide-imm plain-ir intrinsics layerb layerb2"
# Tests that must be rejected (negative fixtures): <test>|<message>.
NEG="operand-placement|illegal PE operand placement
pe-alignment|misaligned PE operand
lone-wait|`wait` must co-issue"
# Tests whose vsm needs --instruction-mode flat on assemble3.
FLAT="peoperand-modes"
# Hazard/race checking is a separate stage: only streams authored to be
# race-free (single-kernel, spaced) belong here. Syntax-only verification
# (checkers disabled) is the stage above.
HAZARD_OK="peoperand-modes l1bml2bm-emit mv-emit mau-emit alu-emit mask-emit imm-fixed-emit lmbar golden golden_mem layerb layerb2"

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
for t in $HAZR; do
  $LLC -mtriple=mncore2 -run-pass=mncore-hazard -mncore-hazard-report $TD/$t.mir -o - 2>&1 \
    | $FC $TD/$t.mir >/dev/null 2>&1 \
    && echo "PASS $t" || { echo "FAIL $t"; fail=$((fail+1)); }
done
for t in $IRL; do
  $LLC -mtriple=mncore2 $TD/$t.ll -o - 2>/dev/null | $FC $TD/$t.ll >/dev/null 2>&1 \
    && echo "PASS $t" || { echo "FAIL $t"; fail=$((fail+1)); }
done
while IFS= read -r spec; do
  t=${spec%%|*}; msg=${spec#*|}
  out=$($LLC -mtriple=mncore2 $TD/$t.mir -o - 2>&1)
  if [ $? -eq 0 ] || ! printf '%s' "$out" | grep -q -- "$msg"; then
    echo "FAIL $t (expected a rejection: $msg)"; fail=$((fail+1))
  else
    echo "PASS $t"
  fi
done <<< "$NEG"

remfail=0
if [ $LOCAL_ONLY = 0 ]; then
  echo "===== assemble3 + gpfn3_package_main ====="
  for t in $EMIT; do
    extra=""
    case " $FLAT " in *" $t "*) extra="$extra --flat" ;; esac
    [ $STRICT = 1 ] && extra="$extra --hazard"
    out=$($TD/scripts/emulate_gpfn3.sh $extra $TD/$t.mir 2>&1)
    rc=$?
    printf '%s\n' "$out" | grep -v '^emulate:'
    if [ $rc -ne 0 ]; then
      if printf '%s' "$out" | grep -q '^SKIP'; then
        if [ $ALLOW_OFFLINE = 0 ]; then
          echo "FAIL  $t  (remote stage unavailable; use --allow-offline)"
          remfail=$((remfail+1))
        fi
      else
        remfail=$((remfail+1))
      fi
    fi
  done
  # IR tests that must also assemble/emulate.
  for t in intrinsics layerb layerb2; do
    out=$($TD/scripts/emulate_gpfn3.sh $TD/$t.ll 2>&1)
    rc=$?
    printf '%s\n' "$out" | grep -v '^emulate:'
    if [ $rc -ne 0 ]; then
      if printf '%s' "$out" | grep -q '^SKIP' && [ $ALLOW_OFFLINE = 1 ]; then
        :
      else
        echo "FAIL  $t  (remote)"; remfail=$((remfail+1))
      fi
    fi
  done
  echo "===== golden numeric (dump + check_dump) ====="
  for t in golden golden_mem; do
    out=$($TD/scripts/emulate_gpfn3.sh --dump $TD/$t.mir 2>&1)
    rc=$?
    printf '%s\n' "$out" | grep -v '^emulate:'
    if [ "$t" = golden ]; then
      exp=('GREG0:n0c0b0m0p0:0=0x2A0000002A' 'GREG1:n0c0b0m0p0:2=0x3900000039')
    else
      exp=('GREG0:n0c0b0m0p0:0=0x39')
    fi
    if [ $rc -ne 0 ] || ! python3 -I $TD/scripts/check_dump.py /tmp/mncore_$t.dmp "${exp[@]}" >/dev/null 2>&1; then
      echo "FAIL  $t  (golden value)"; remfail=$((remfail+1))
    fi
  done
  echo "===== assemble3 hazard/race checker (separate stage) ====="
  for t in $HAZARD_OK; do
    case " $FLAT " in *" $t "*) hz="--flat" ;; *) hz="" ;; esac
    src=$TD/$t.mir; [ -f "$src" ] || src=$TD/$t.ll
    out=$($TD/scripts/verify_assemble3.sh $hz $src 2>&1)
    rc=$?
    printf '%s\n' "$out" | grep -v '^assemble3 verify:'
    [ $rc -ne 0 ] && { echo "FAIL  $t  (hazard/race)"; remfail=$((remfail+1)); }
  done
  echo "remote failures: $remfail"
fi
total=$((fail + remfail))
echo "===== done (local failures: $fail, remote failures: $remfail) ====="
[ $total -eq 0 ]
