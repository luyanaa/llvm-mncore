#!/usr/bin/env python3
"""Check the block-float golden (`golden_mau_bfn.mir`) against the SDM model.

`emulate_gpfn3.sh --dump golden_mau_bfn.mir` writes a dump; this script
recomputes every PE longword of the dump window from the input pattern and
the SDM 4.4 block-float conversion, and compares bit for bit.

Scope: `fbfn` (single precision, block = 4 words = one 32-bit word from each
of the four PEs of a MAB). The two 32-bit words of a PE longword are two
independent blocks (SDM 3.6.12.12); the longword access is 4 cycles, one
longword per cycle (throughput statement in 3.6.12.12), so one instruction
converts 4 longwords.

Measured model (gpfn3, 2026-11-04; design doc section 22.8):
  cycle c reads the per-PE longword at word (src_addr + 2c), i.e. longword
  (src_addr/2 + c), and writes longword (dst_addr/2 + c). The conversion is
  the SDM 4.4 procedure below, applied per word lane across the four PEs of
  each MAB.

BF encoding (SDM 4.4): sign and exponent fields are the source IEEE-754
fields; the 23-bit mantissa field has an explicit leading one, i.e. the bit
pattern is interpreted as (-1)^s * 2^(e-127) * sum_{j=1..23} 2^(1-j) M_j.
So BF(1.0) with common exponent 0 is 0x3FC00000, which IEEE reads as 1.5 --
the raw bits are never IEEE values.

Conversion of a 4-word block (n = 4, m = 23):
  1. common exponent = max exponent field
  2. +1 if an element at the max exponent has an all-ones mantissa
  3. common >= 0xFF -> every output is infinity with the input sign
     an input with exponent 0 -> output exponent = common, mantissa 0
     all inputs with exponent 0 -> every output is all zeros
  4. per element, the 24-bit significand (hidden 1 + mantissa) is shifted
     right by (common - e + 1) and rounded to nearest even; underflow
     (rounded mantissa 0) keeps the common exponent

Usage
  check_mau_bfn.py <dump> [--verbose]

Exit status is 0 only when every cell of the dump window matches.
"""

import re
import sys

LINE = re.compile(
    r"DEBUG-GREG0\(n(?P<n>\d+)c(?P<c>\d+)b(?P<b>\d+)m(?P<mab>\d+)p(?P<pe>\d+),"
    r"(?P<idx>\d+)\):.*?\bv:(?P<val>0x[0-9a-fA-F]+)")

# --- input pattern (mirrored in golden_mau_bfn.mir) -----------------------

def shift_exp(v, k):
    """Add k to the exponent field of a normal f32 (zero/subnormal untouched)."""
    e = (v >> 23) & 0xFF
    if e == 0:
        return v
    return (v & 0x807FFFFF) | (((e + k) & 0xFF) << 23)

UNITS = [0x3F800000, 0x40000000, 0x40400000, 0x40800000]  # 1, 2, 3, 4
LOWZ = [0x41000000, 0x3F000000, 0x00000000, 0x3F800001]  # 8, .5, 0, 1+2^-23
CARRY = [0x3FFFFFFF, 0x3F800000, 0x3F800000, 0x3F800000]  # 1.999..., 1,1,1
INF = [0x7F800000, 0xC0000000, 0x3FC00000, 0x00000001]   # inf, -2, 1.5, sub
TIES = [0x3F800001, 0x3F800002, 0x3F800003, 0x3F800000]  # RNE ties

def longword(plane, mab, q):
    """The per-PE longword loaded by the golden for (plane, MAB, local PE)."""
    if mab == 0:
        hi, lo = UNITS[q], LOWZ[q]
    elif mab == 1:
        hi, lo = 0x3F800000, CARRY[q]
    elif mab == 2:
        hi, lo = 0x00000000, INF[q]
    elif mab == 3:
        hi, lo = TIES[q], TIES[q]
    else:
        t = plane * 16 + mab * 4 + q
        hi = 0x3F800000 | ((t & 0xFF) << 8)
        lo = 0x40000000 | ((t & 0x7FF) << 8)
    return (shift_exp(hi, plane) << 32) | shift_exp(lo, plane)

# --- SDM 4.4 block-float conversion --------------------------------------

def rne_shift(x, sh):
    """round(x / 2**sh) to nearest, ties to even, x >= 0 integer."""
    q, r = divmod(x, 1 << sh)
    half = 1 << (sh - 1)
    if r < half:
        return q
    if r > half:
        return q + 1
    return q + (q & 1)

def bf32_block(vals):
    """4 IEEE-754 f32 bit patterns -> 4 BF-encoded bit patterns."""
    s = [(v >> 31) & 1 for v in vals]
    e = [(v >> 23) & 0xFF for v in vals]
    f = [v & 0x7FFFFF for v in vals]
    if all(x == 0 for x in e):
        return [s[i] << 31 for i in range(4)]
    cm = max(e)
    if any(e[i] == cm and f[i] == 0x7FFFFF for i in range(4)):
        cm += 1
    if cm >= 0xFF:
        return [(s[i] << 31) | (0xFF << 23) for i in range(4)]
    out = []
    for i in range(4):
        if e[i] == 0:
            out.append((s[i] << 31) | (cm << 23))
            continue
        m = rne_shift(0x800000 | f[i], cm - e[i] + 1)
        out.append((s[i] << 31) | (cm << 23) | (m & 0x7FFFFF))
    return out

# --- window ---------------------------------------------------------------

NPLANES = 4     # four cycles -> four input longwords, four output longwords
NWORDS = 16     # d get $lr0 32 (16 longwords): inputs 0..3, outputs 4..7

def expect():
    exp = {}
    for mab in range(16):
        for pl in range(NPLANES):
            his = [longword(pl, mab, q) >> 32 for q in range(4)]
            los = [longword(pl, mab, q) & 0xFFFFFFFF for q in range(4)]
            bh = bf32_block(his)
            bl = bf32_block(los)
            for q in range(4):
                exp[(mab, q, pl)] = longword(pl, mab, q)
                exp[(mab, q, NPLANES + pl)] = (bh[q] << 32) | bl[q]
    return exp

def parse(path):
    """Key: (group, l2b, l1b, mab, local PE, longword).

    The dump spans every (group, l2b, l1b) unit -- the kernel is SPMD and
    runs on all of them -- so the unit coordinates must be part of the key.
    Dropping them would let a wrong value in one unit be overwritten by a
    correct one from another (that is exactly how a one-bit mutation of this
    dump slipped through once); the golden therefore checks every unit.
    """
    out = {}
    for line in open(path):
        m = LINE.search(line)
        if m:
            # the dump labels a longword by its word address (2 * longword)
            out[(int(m.group("n")), int(m.group("c")), int(m.group("b")),
                 int(m.group("mab")), int(m.group("pe")),
                 int(m.group("idx")) // 2)] = int(m.group("val"), 16)
    return out

def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    dump = argv[1]
    verbose = "--verbose" in argv[2:]
    exp = expect()
    got = parse(dump)
    if not got:
        print("check_mau_bfn: no GREG0 entries in", dump)
        return 1
    units = sorted({key[:3] for key in got})
    if not units:
        print("check_mau_bfn: no units in", dump)
        return 1

    bad = []
    for unit in units:
        for (mab, q, lw), want in exp.items():
            have = got.get(unit + (mab, q, lw))
            if have != want:
                bad.append((unit + (mab, q, lw), want, have))
        # nothing may be written anywhere else in the window
        for key, have in got.items():
            if key[:3] != unit or key[5] >= NWORDS:
                continue
            if key[3:] not in exp and have != 0:
                bad.append((key, 0, have))
        # the dump must cover the whole window (a truncated dump is a failure)
        for mab in range(16):
            for q in range(4):
                for lw in range(NWORDS):
                    if unit + (mab, q, lw) not in got:
                        bad.append((unit + (mab, q, lw), "present", None))

    if verbose:
        for key in sorted(exp):
            print(key, hex(exp[key]))
    if bad:
        print("check_mau_bfn: %d mismatches" % len(bad))
        for key, want, have in bad[:20]:
            w = want if isinstance(want, str) else hex(want)
            h = have if have is None else hex(have)
            print("  n%dc%db%d mab%-2d pe%d lw%-2d: want %s got %s"
                  % (key[0], key[1], key[2], key[3], key[4], key[5], w, h))
        return 1
    print("check_mau_bfn: OK (%d cells x %d units over 64 PEs)"
          % (len(exp), len(units)))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
