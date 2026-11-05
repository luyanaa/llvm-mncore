#!/usr/bin/env python3
"""Check the G0 matrix-transport golden (`golden_mau_g0*.mir`) against the
measured MAU model.

`emulate_gpfn3.sh --dump golden_mau_g0.mir` writes a dump; this script
recomputes every PE longword of the dump window from the model below and
compares. It is a *model* check, not a self-consistency check: the
expectations follow the placement measured on gpfn3 (2026-11-04), so a
change in either the emulator or the encoder that moves a matrix cell
fails here.

Measured model (f32, `fmwrite`/`fmread` at longword width; SDM 3.6.10/3.6.11
semantics with the placement taken from the dump, see design doc 22.7):

  fmwrite  <src> $l(x|y)<row>
      cycle c = 0..3 reads the per-PE longword at word (src_addr + 2c) and
      writes matrix row (row + c) mod 8; within the row, PE q's longword
      LOW  half goes to column 2q and its HIGH half to column 2q+1.
      The matrix of a MAB is fed only by that MAB's own four PEs (q), so
      the content is the same in every MAB.

  fmread   $l(x|y)<col> <dst>
      cycle c = 0..3 reads matrix column (col + c); PE q of every MAB
      receives rows 2q (HIGH word) and 2q+1 (LOW word) of that column,
      stored in the longword at word (dst_addr/2 + (c XOR 1)).

The `c XOR 1` on the destination is the one non-obvious part: within each
pair of cycles the destination longwords are swapped. It was measured (not
assumed): with a read start column of 0 the odd columns land in the even
destination longwords and vice versa, and starting at column 1 shifts the
whole pattern by one column, which no plain-stride model reproduces.

Usage
  check_mau_g0.py <dump> [--case full|y] [--verbose]

Exit status is 0 only when every cell of the dump window matches.
"""

import re
import sys

LINE = re.compile(
    r"DEBUG-GREG0\(n(?P<n>\d+)c(?P<c>\d+)b(?P<b>\d+)m(?P<mab>\d+)p(?P<pe>\d+),"
    r"(?P<idx>\d+)\):.*?\bv:(?P<val>0x[0-9a-fA-F]+)")

# The 64-bit source pattern per (slot, local PE). Slots 0..3 become matrix
# rows 0..3 (set A), slots 4..7 rows 4..7 (set B).
def src_longword(slot, p):
    if slot < 4:
        return ((0x20 + slot) << 32) | (0x10 + p)
    return ((0x40 + slot) << 32) | (0x30 + p)


def matrix_row(slot, mab):
    """Matrix row `slot` of MAB `mab` as [column] -> 32-bit word.

    A matrix register belongs to one MAB and is fed only by that MAB's own
    four PEs: column j <- local PE j/2, word j%2. The source pattern is
    written per global PE, so MAB m reads global PE (4m + j/2), which makes
    the row content differ from MAB to MAB (and is how the dump proves the
    matrix is per-MAB rather than shared).
    """
    row = [0] * 8
    for q in range(4):
        lw = src_longword(slot, 4 * mab + q)
        row[2 * q] = lw & 0xFFFFFFFF
        row[2 * q + 1] = (lw >> 32) & 0xFFFFFFFF
    return row


def expect_full():
    """golden_mau_g0.mir: two writes (rows 0-3, 4-7), two reads (cols 0-3, 4-7)."""
    exp = {}
    for mab in range(16):
        rows = [matrix_row(k, mab) for k in range(8)]
        for q in range(4):
            # source longwords stay in place
            for slot in range(8):
                exp[(mab, q, slot)] = src_longword(slot, 4 * mab + q)
            for base_lw, start_col in ((16, 0), (20, 4)):
                for c in range(4):
                    col = start_col + c
                    hi = rows[2 * q][col]
                    lo = rows[2 * q + 1][col]
                    exp[(mab, q, base_lw + (c ^ 1))] = (hi << 32) | lo
    return exp, 64


def expect_y():
    """golden_mau_g0_y.mir: one write of rows 0-3, one read of column 0."""
    exp = {}
    for mab in range(16):
        # only rows 0..3 are written by this golden; 4..7 stay zero
        rows = [matrix_row(k, mab) for k in range(4)] + [[0] * 8 for _ in range(4)]
        for q in range(4):
            for slot in range(4):
                exp[(mab, q, slot)] = src_longword(slot, 4 * mab + q)
            for c in range(4):
                hi = rows[2 * q][c]
                lo = rows[2 * q + 1][c]
                exp[(mab, q, 16 + (c ^ 1))] = (hi << 32) | lo
    return exp, 48


def parse(path):
    """Key: (group, l2b, l1b, mab, local PE, longword).

    The dump spans every (group, l2b, l1b) unit -- the kernel is SPMD and
    runs on all of them -- so the unit coordinates must be part of the key.
    Dropping them lets a wrong value in one unit be overwritten by a correct
    one from another unit with the same (mab, pe, longword).
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
    case = "full"
    verbose = False
    i = 2
    while i < len(argv):
        if argv[i] == "--case":
            case = argv[i + 1]
            i += 2
        elif argv[i] == "--verbose":
            verbose = True
            i += 1
        else:
            print("unknown argument", argv[i])
            return 2
    if case == "full":
        exp, nwords = expect_full()
    elif case == "y":
        exp, nwords = expect_y()
    else:
        print("unknown case", case)
        return 2

    got = parse(dump)
    if not got:
        print("check_mau_g0: no GREG0 entries in", dump)
        return 1
    units = sorted({key[:3] for key in got})
    if not units:
        print("check_mau_g0: no units in", dump)
        return 1

    bad = []
    for unit in units:
        # 1. every modelled cell
        for (mab, q, lw), want in exp.items():
            have = got.get(unit + (mab, q, lw))
            if have != want:
                bad.append((unit + (mab, q, lw), want, have))
        # 2. nothing may be written anywhere else in the window
        for key, have in got.items():
            if key[:3] != unit or key[5] >= nwords:
                continue
            if key[3:] not in exp and have != 0:
                bad.append((key, 0, have))
        # 3. the dump must cover the whole window (a truncated dump fails)
        for mab in range(16):
            for q in range(4):
                for lw in range(nwords):
                    if unit + (mab, q, lw) not in got:
                        bad.append((unit + (mab, q, lw), "present", None))

    if verbose:
        for key in sorted(exp):
            print(key, hex(exp[key]))
    if bad:
        print("check_mau_g0: %d mismatches (case %s)" % (len(bad), case))
        for key, want, have in bad[:20]:
            w = want if isinstance(want, str) else hex(want)
            h = have if have is None else hex(have)
            print("  n%dc%db%d mab%-2d pe%d lw%-2d: want %s got %s"
                  % (key[0], key[1], key[2], key[3], key[4], key[5], w, h))
        return 1
    print("check_mau_g0: OK (case %s, %d cells x %d units over 64 PEs)"
          % (case, len(exp), len(units)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
