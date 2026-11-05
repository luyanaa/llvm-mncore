#!/bin/bash
# Top-level MNCore regression: local FileCheck + remote assemble3 + emulator.
# Run from anywhere; paths are absolute.
#
#   run_regression.sh                 # everything; non-zero on ANY failure
#   run_regression.sh --local         # FileCheck only
#   run_regression.sh --allow-offline # FileCheck + remote if reachable;
#                                      # an unreachable ssh is not a failure
#   run_regression.sh --strict        # add assemble3 --hazard on remote runs
#
# The "Program container schema" stage is pure Python over committed mnclc
# fixtures, so it runs in every mode including --local.
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
EMIT="peoperand-modes l1bml2bm-emit mv-emit mau-emit alu-emit mask-emit imm-fixed-emit lmbar golden golden_mem golden_store golden_l1bmm golden_l1bmd golden_l1bmd_store golden_roundtrip golden_mau_smoke"
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
# Matrix-transport goldens: FileCheck + the MAU placement model checker
# (scripts/check_mau_g0.py), run as their own numeric stage.
MAU="golden_mau_g0 golden_mau_g0_y"
# Block-float goldens: FileCheck + the SDM 4.4 conversion model
# (scripts/check_mau_bfn.py), run as their own numeric stage.
BFN="golden_mau_bfn"
# G2 tripwire: pins the measured absence of the MAU product on gpfn3
# (design doc 22.9). It is a negative result, so it runs its own stage.
G2TRAP="golden_mau_g2"
# Tests whose vsm needs --instruction-mode flat on assemble3.
FLAT="peoperand-modes"
# Hazard/race checking is a separate stage: only streams authored to be
# race-free (single-kernel, spaced) belong here. Syntax-only verification
# (checkers disabled) is the stage above.
HAZARD_OK="peoperand-modes l1bml2bm-emit mv-emit mau-emit alu-emit mask-emit imm-fixed-emit lmbar golden golden_mem golden_store golden_l1bmm golden_l1bmd golden_l1bmd_store golden_roundtrip golden_mau_smoke golden_mau_g0 golden_mau_g0_y golden_mau_bfn golden_mau_g2 l1load-dist l1store-dist layerb layerb2"

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
for t in $MAU $BFN $G2TRAP; do
  $LLC -mtriple=mncore2 $TD/$t.mir -o - 2>/dev/null | $FC $TD/$t.mir >/dev/null 2>&1 \
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

# Program container: mnclc-produced mncore::Program fixtures, decoded and
# pinned field by field. Fully local (pure Python, no mncl, no sde64), so it
# runs even offline. These pin the FlatBuffers schema recovered in
# docs/mncore_mnir_offload.md §6 A.1 -- the same numbers must survive any
# future Program *writer*.
echo "===== Program container schema ====="
PR=$TD/Inputs/prog
progchk() {
  t=$1; shift
  if python3 -I $TD/scripts/check_prog.py "$@" >/dev/null 2>&1; then
    echo "PASS prog:$t"
  else
    echo "FAIL prog:$t"; fail=$((fail+1))
  fi
}
progchk ke    $PR/ke.bin    size=256  name=k nargs=0 nfixup=0 L=192 blk=0    tail=64
progchk s28   $PR/s28.bin   size=512  name=k nargs=2 nfixup=1 L=192 blk=256  tail=64  tags=1,1 fx=0/0/0
progchk gen26 $PR/gen26.bin size=1088 name=k nargs=2 nfixup=3 L=256 blk=768  tail=64  tags=1,1 fx=0/0/0,2/0/0,4/1/0
progchk gen16 $PR/gen16.bin size=2048 name=k nargs=7 nfixup=6 L=320 blk=1536 tail=192 tags=1,1,1,1,1,1,1 fx=0/0/0,2/1/0,4/2/0,6/3/0,8/4/0,10/5/0
# Negative fixture: a deliberately wrong expectation must be rejected.  If it
# is accepted, the checker stopped comparing and the four PASSes above mean
# nothing.
if python3 -I $TD/scripts/check_prog.py $PR/gen16.bin nargs=6 >/dev/null 2>&1; then
  echo "FAIL prog:negative (wrong nargs accepted)"; fail=$((fail+1))
else
  echo "PASS prog:negative"
fi
# Negative fixture: a file that is not a Program must be rejected cleanly and
# without a huge allocation.  A failed mnclc leaves text behind, and trusting
# its vector lengths once drove the decoder into MemoryError.
tmpp=$(mktemp "${TMPDIR:-/tmp}/progneg.XXXXXX")
head -c 200 /dev/zero | tr '\0' '0' > $tmpp
if python3 -I $TD/scripts/mnprog.py --check $tmpp >/dev/null 2>&1; then
  echo "FAIL prog:not-a-program accepted"; fail=$((fail+1))
else
  echo "PASS prog:not-a-program"
fi
rm -f $tmpp

# Program rebuild: reconstruct each fixture from its decoded fields with a
# from-scratch 64-bit FlatBuffers writer and require byte identity.  This is
# the completeness proof for the schema: every header byte -- creation order,
# the instblock 64-byte alignment, vtable sharing -- has to come out right.
echo "===== Program rebuild (byte-identical) ====="
progrebuild() {
  t=$1
  if python3 -I $TD/scripts/mnprog_build.py $PR/$t.bin | grep -q '^IDENTICAL'; then
    echo "PASS rebuild:$t"
  else
    echo "FAIL rebuild:$t"; fail=$((fail+1))
  fi
}
for t in ke s28 gen26 gen16; do progrebuild $t; done
# Negative fixture: truncate the last 8 bytes of s28 (its re-derived tail) and
# the rebuild must fail -- the rebuilt file is 512 B, the input 504 B.  If this
# passes, the comparison is not actually comparing.
tmpp=$(mktemp "${TMPDIR:-/tmp}/progneg.XXXXXX")
head -c 504 $PR/s28.bin > $tmpp
if python3 -I $TD/scripts/mnprog_build.py $tmpp >/dev/null 2>&1; then
  echo "FAIL rebuild:negative (modified file accepted)"; fail=$((fail+1))
else
  echo "PASS rebuild:negative"
fi
rm -f $tmpp

remfail=0
if [ $LOCAL_ONLY = 0 ]; then
  # One ssh connection for the whole run. Each test opens ~5 connections on
  # suzhou; a burst of ~200 in a couple of minutes gets the host to close new
  # connections ("Connection closed by ... port 57953"), which used to be
  # reported as test failures. A shared master keeps the connection count at
  # one and makes those failures impossible by construction.
  export MN_SSH_CONTROL="/tmp/mncore_ssh_$$.sock"
  ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=yes \
      -o ControlPath="$MN_SSH_CONTROL" -o ControlPersist=300 -N -f suzhou \
      2>/dev/null || true
  trap 'ssh -o ControlPath="$MN_SSH_CONTROL" -O exit suzhou >/dev/null 2>&1 || true' EXIT
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
  echo "===== golden MAU G0 (matrix placement model) ====="
  for t in $MAU; do
    case $t in
      golden_mau_g0) c=full ;;
      golden_mau_g0_y) c=y ;;
    esac
    out=$($TD/scripts/emulate_gpfn3.sh --dump $TD/$t.mir 2>&1)
    rc=$?
    printf '%s\n' "$out" | grep -v '^emulate:'
    if [ $rc -ne 0 ]; then
      if printf '%s' "$out" | grep -q '^SKIP' && [ $ALLOW_OFFLINE = 1 ]; then
        :
      else
        echo "FAIL  $t  (remote)"; remfail=$((remfail+1))
      fi
    elif ! python3 -I $TD/scripts/check_mau_g0.py /tmp/mncore_$t.dmp --case $c; then
      echo "FAIL  $t  (MAU G0 model)"; remfail=$((remfail+1))
    fi
  done
  echo "===== golden block float G1 (SDM 4.4 conversion model) ====="
  for t in $BFN; do
    out=$($TD/scripts/emulate_gpfn3.sh --dump $TD/$t.mir 2>&1)
    rc=$?
    printf '%s\n' "$out" | grep -v '^emulate:'
    if [ $rc -ne 0 ]; then
      if printf '%s' "$out" | grep -q '^SKIP' && [ $ALLOW_OFFLINE = 1 ]; then
        :
      else
        echo "FAIL  $t  (remote)"; remfail=$((remfail+1))
      fi
    elif ! python3 -I $TD/scripts/check_mau_bfn.py /tmp/mncore_$t.dmp; then
      echo "FAIL  $t  (block float model)"; remfail=$((remfail+1))
    fi
  done
  echo "===== G2 tripwire (gpfn3 fmfma does not compute A*x) ====="
  for t in $G2TRAP; do
    out=$($TD/scripts/emulate_gpfn3.sh --dump $TD/$t.mir 2>&1)
    rc=$?
    printf '%s\n' "$out" | grep -v '^emulate:'
    if [ $rc -ne 0 ]; then
      if printf '%s' "$out" | grep -q '^SKIP' && [ $ALLOW_OFFLINE = 1 ]; then
        :
      else
        echo "FAIL  $t  (remote)"; remfail=$((remfail+1))
      fi
    elif ! python3 -I $TD/scripts/check_mau_g2.py /tmp/mncore_$t.dmp; then
      echo "FAIL  $t  (G2 tripwire: emulator changed, re-open G2)"
      remfail=$((remfail+1))
    fi
  done
  echo "===== golden numeric (dump + check_dump) ====="
  for t in golden golden_mem golden_store golden_l1bmm golden_l1bmd golden_l1bmd_store golden_roundtrip; do
    out=$($TD/scripts/emulate_gpfn3.sh --dump $TD/$t.mir 2>&1)
    rc=$?
    printf '%s\n' "$out" | grep -v '^emulate:'
    if [ "$t" = golden ]; then
      exp=('GREG0:n0c0b0m0p0:0=0x2A0000002A' 'GREG1:n0c0b0m0p0:2=0x3900000039')
    elif [ "$t" = golden_mem ]; then
      exp=('GREG0:n0c0b0m0p0:0=0x39')
    elif [ "$t" = golden_store ]; then
      exp=('L1BM:*:0=0x3900000039')
    elif [ "$t" = golden_l1bmm ]; then
      exp=('GREG0:n0c0b0m0p0:0=0x1' 'GREG0:n0c0b0m0p1:0=0x2'
           'GREG0:n0c0b0m0p2:0=0x3' 'GREG0:n0c0b0m0p3:0=0x4'
           'GREG0:n0c0b0m1p0:0=0x1' 'GREG0:n0c0b0m15p3:0=0x4')
    elif [ "$t" = golden_l1bmd ]; then
      exp=('GREG0:*:0=0xaa' 'GREG0:*:4=0xbb')
    elif [ "$t" = golden_l1bmd_store ]; then
      exp=('L1BM:*:0=0x3900000039')
    elif [ "$t" = golden_roundtrip ]; then
      # Planes 0 and 64 of the roundtrip fixture hold different constants;
      # the entries are keyed by L1BM block id, so both are checked by index.
      exp=('L1BM:*:0=0x4000000040000000' 'L1BM:*:64=0x4080000040800000')
    else
      exp=('L1BM:*:0=0x4080000040800000')
    fi
    # Distinguish "the emulation did not run" (transport/e.g. a dropped ssh)
    # from "it ran and the values are wrong": both fail, but a numeric
    # FAIL should never be reported for an infrastructure hiccup.
    if [ $rc -ne 0 ]; then
      echo "FAIL  $t  (remote)"; remfail=$((remfail+1)); continue
    fi
    if ! python3 -I $TD/scripts/check_dump.py /tmp/mncore_$t.dmp "${exp[@]}" >/dev/null 2>&1; then
      echo "FAIL  $t  (golden value)"; remfail=$((remfail+1))
    fi
    if [ "$t" = golden_l1bmd_store ]; then
      for plane in '$lb0' '$lb64'; do
        if ! python3 -I $TD/scripts/check_dump.py --contains /tmp/mncore_$t.dmp "0x3900000039) #d get $plane 1" >/dev/null 2>&1; then
          echo "FAIL  $t  (plane $plane)"; remfail=$((remfail+1))
        fi
      done
    fi
    if [ "$t" = golden_roundtrip ]; then
      if ! python3 -I $TD/scripts/check_dump.py --contains /tmp/mncore_$t.dmp "0x4000000040000000) #d get \$lb0 1" >/dev/null 2>&1; then
        echo "FAIL  $t  (low plane)"; remfail=$((remfail+1))
      fi
      if ! python3 -I $TD/scripts/check_dump.py --contains /tmp/mncore_$t.dmp "0x4080000040800000) #d get \$lb64 1" >/dev/null 2>&1; then
        echo "FAIL  $t  (high plane)"; remfail=$((remfail+1))
      fi
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
