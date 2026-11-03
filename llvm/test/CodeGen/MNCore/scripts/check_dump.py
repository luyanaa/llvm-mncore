#!/usr/bin/env python3
"""Verify values in a gpfn3_package_main `-d <file>` dump.

`d get` output lines look like (they may be interleaved with other text):

    DEBUG-GREG0(n0c0b0m0p0,0):(f:0, i:{{0x0,0x0},{0x0,0x0}}, v:0x0)
    DEBUG-LM1(n0c0b0m0p0,0):(1.5, 1.5, 1.5, 1.5) (0x3f00, ...) #d geth ...

check_dump.py parses the `v:<hex>` (integer/longword interpretation), which
d get prints for every memory element regardless of <dtype>.

Usage
  check_dump.py <dump> <expect>...
      <expect> := <MEM>:<ids>:<index>=<value>
      e.g.      GREG0:n0c0b0m0p0:0=0x1   or   GREG0:*:0=0x1
  check_dump.py --list <dump>          print every parsed entry
  check_dump.py --contains <dump> <txt>  dump must contain a literal string
  check_dump.py --selftest             validate the parser on a sample

Exit status is 0 only when every expectation matches. `*` matches any
hierarchy-id string; the index is the <num_of_words> position per entry.

NOTE: authored while `ssh suzhou` was unreachable, so the sample in
--selftest is synthetic. Validate once against a real dump with
`emulate_gpfn3.sh --dump <test.mir>` before relying on it in CI.
"""

import re
import sys

LINE = re.compile(
    r"DEBUG-(?P<mem>[A-Za-z0-9]+)\((?P<ids>[^,]+),(?P<idx>\d+)\):"
    r".*?\bv:(?P<val>0x[0-9a-fA-F]+)")

SAMPLE = """\
DEBUG-GREG0(n0c0b0m0p0,0):(f:0, i:{{0x0,0x0},{0x0,0x0}}, v:0x0) #d get $lr0n0c0b0m0p0 1
DEBUG-GREG0(n0c0b0m0p0,1):(f:0, i:{{0x0,0x0},{0x0,0x1}}, v:0x1) #d get $lr0n0c0b0m0p0 1
DEBUG-LM1(n0c0b0m0p0,0):(f:1.5, i:{{0x3f00,0x3f00},{0x3f00,0x3f00}}, v:0x3f003f003f003f00) #d get $ln0n0c0b0m0p0 1
"""


def parse(text):
    out = {}
    for line in text.splitlines():
        m = LINE.search(line)
        if m:
            key = (m.group("mem"), m.group("ids"), int(m.group("idx")))
            out[key] = int(m.group("val"), 16)
    return out


def selftest():
    d = parse(SAMPLE)
    assert d[("GREG0", "n0c0b0m0p0", 0)] == 0
    assert d[("GREG0", "n0c0b0m0p0", 1)] == 1
    assert d[("LM1", "n0c0b0m0p0", 0)] == 0x3F003F003F003F00
    return True


def main(argv):
    if len(argv) >= 2 and argv[1] == "--selftest":
        selftest()
        print("check_dump: selftest OK")
        return 0
    if len(argv) >= 3 and argv[1] == "--list":
        for (mem, ids, idx), val in sorted(parse(open(argv[2]).read()).items()):
            print(f"{mem}:{ids}:{idx}=0x{val:x}")
        return 0
    if len(argv) >= 4 and argv[1] == "--contains":
        text = open(argv[2]).read()
        ok = argv[3] in text
        print(("OK   " if ok else "MISS ") + f"contains {argv[3]!r}")
        return 0 if ok else 1
    if len(argv) < 3:
        print(__doc__)
        return 2

    d = parse(open(argv[1]).read())
    bad = 0
    for spec in argv[2:]:
        try:
            lhs, rhs = spec.split("=", 1)
            mem, ids, idx = lhs.split(":")
            want = int(rhs, 0)
        except ValueError:
            print(f"BAD SPEC {spec!r} (want MEM:IDS:INDEX=VALUE)")
            bad += 1
            continue
        if ids == "*":
            got = sorted((k, v) for k, v in d.items()
                         if k[0] == mem and k[2] == int(idx))
            if not got:
                print(f"FAIL {spec}: no {mem} entry with index {idx}")
                bad += 1
            for k, v in got:
                if v != want:
                    print(f"FAIL {k[0]}:{k[1]}:{k[2]} = 0x{v:x}, want 0x{want:x}")
                    bad += 1
                else:
                    print(f"OK   {k[0]}:{k[1]}:{k[2]} = 0x{v:x}")
            continue
        key = (mem, ids, int(idx))
        if key not in d:
            print(f"FAIL {spec}: entry not found in dump")
            bad += 1
        elif d[key] != want:
            print(f"FAIL {spec}: got 0x{d[key]:x}, want 0x{want:x}")
            bad += 1
        else:
            print(f"OK   {spec}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
