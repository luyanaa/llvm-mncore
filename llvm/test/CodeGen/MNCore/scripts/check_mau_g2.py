#!/usr/bin/env python3
"""Check the G2 tripwire (`golden_mau_g2.mir`).

This is deliberately the opposite of a numeric golden. It pins the *measured
absence* of the MAU product on gpfn3 (design doc section 22.9):

  gpfn3 writes `dst := y` for `fmfma`; the A*x term contributes nothing.

The fixture feeds operands where SDM 3.6.9.4 predicts a result that cannot be
confused with y:

  A rows 0..3 = BF(1.0)   x = 4 x BF(1.0)   y = 0x40C00000 (IEEE 6.0)
  predicted  dst = 1+1+1+1 + 6.0 = 10.0 -> 0x41200000
  measured   dst = y bit for bit       -> 0x40C00000

Exit status:
  0  the measured pattern still holds  -> the G2 gap is still open
  1  the dump does not match          -> the emulator changed; re-open G2
     (either the product is now computed, or the transport changed as well,
     in which case golden_mau_g2.mir must be re-derived from measurement)

The message makes the two cases distinguishable: a mismatch at the dst words
alone means the product appeared; a mismatch anywhere means the transport
model moved too.

Usage
  check_mau_g2.py <dump> [--verbose]
"""

import re
import sys

LINE = re.compile(
    r"DEBUG-(?P<mem>GREG[01])\(n(?P<n>\d+)c(?P<c>\d+)b(?P<b>\d+)m(?P<mab>\d+)"
    r"p(?P<pe>\d+),(?P<idx>\d+)\):.*?\bv:(?P<val>0x[0-9a-fA-F]+)")

# --- measured transport model (design doc 22.7, 22.8, 22.9) ---------------

BF1 = 0x3FC00000                       # BF(1.0): explicit leading one at bit 22
# The fixture's y words. Read as IEEE f32 (fmfma's addend is plain f32, see
# vendor header mncore2.h and design doc 22.10) this is 6.0; under the block
# float convention it is 4.0. Either way the product is missing, so the value
# is quoted as a bit pattern everywhere.
YVAL = 0x40C00000
PROD = 0x41200000                      # 10.0: A*x (4.0) + y (6.0)

WORDS = 64                             # `d get $lr0 64` / `$ls0 64`

# Longword-indexed expectations. Everything not listed is zero.
GRF0_LW = {
    0: (YVAL << 32) | YVAL,            # dst cycle 0 <- y lw0
    1: (YVAL << 32) | YVAL,            # dst cycle 1 <- y lw1
    2: 0,                              # dst cycle 2 <- y lw2 (beyond the block)
    3: 0,                              # dst cycle 3 <- y lw3
    4: (BF1 << 32) | BF1,              # x block
    5: (BF1 << 32) | BF1,
    6: (YVAL << 32) | YVAL,            # y block (source of the PASSA copy)
    7: (YVAL << 32) | YVAL,
}
GRF1_LW = {
    6: (YVAL << 32) | YVAL,            # PASSA_l copy of the y block
    7: (YVAL << 32) | YVAL,
}

DST_LW = (0, 1, 2, 3)


def expected(grf):
    table = GRF0_LW if grf == 0 else GRF1_LW
    out = []
    for lw in range(WORDS // 2):
        out.append(table.get(lw, 0))
    return out


def parse(path):
    """(grf, mab, pe) -> list of 64 word values, and the longword view."""
    words = {}
    with open(path) as f:
        for line in f:
            m = LINE.match(line)
            if not m:
                continue
            key = (int(m.group("mem")[-1]), int(m.group("mab")),
                   int(m.group("pe")))
            words.setdefault(key, {})[int(m.group("idx"))] = int(
                m.group("val"), 16)
    return words


def main(argv):
    if len(argv) < 1:
        sys.stderr.write(__doc__)
        return 2
    verbose = "--verbose" in argv
    dump = argv[0]
    data = parse(dump)
    if not data:
        sys.stderr.write("check_mau_g2: no GREG lines in %s\n" % dump)
        return 2

    problems = []
    dst_only = True
    checked = 0
    for (grf, mab, pe), vals in sorted(data.items()):
        want = expected(grf)
        for lw in range(WORDS // 2):
            got_lo = vals.get(2 * lw, 0)
            got_hi = vals.get(2 * lw + 1, 0)
            got = (got_hi << 32) | got_lo
            if got == want[lw]:
                continue
            checked += 1
            watched = grf == 0 and lw in DST_LW
            if not watched:
                dst_only = False
            if verbose or len(problems) < 8:
                problems.append(
                    "  GRF%d m%d p%d longword %2d: got 0x%016x want 0x%016x%s"
                    % (grf, mab, pe, lw, got, want[lw],
                       "  <- dst word" if watched else ""))
    if not problems:
        print("check_mau_g2: OK -- gpfn3 still writes dst := y "
              "(product is not computed); G2 stays open")
        return 0

    print("check_mau_g2: MISMATCH -- the measured model of gpfn3 changed.")
    print("\n".join(problems))
    if dst_only:
        print("Only the fmfma destination words moved. If the new dst words "
              "are 0x%08x\n(10.0 = A*x + y) the emulator now computes the "
              "product: close G2 and turn\nthis fixture into a numeric golden "
              "(design doc section 22.9)." % PROD)
    else:
        print("The change is not confined to the fmfma destination, so the "
              "transport model\nitself moved; re-derive golden_mau_g2.mir "
              "from a fresh measurement.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
