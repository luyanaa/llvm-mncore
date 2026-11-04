#!/usr/bin/env python3
"""Pin the decoded contents of a mncore::Program container.

Companion to mnprog.py: it re-decodes the file and compares only the fields
named on the command line, so a fixture pins exactly what it was chosen to
pin (layout split, argument tags, fixup triplets) while leaving the rest free.

    check_prog.py <prog.bin> key=value [key=value ...]

    size=2048  name=k  nargs=7  nfixup=6  L=320  blk=1536  tail=192
    tags=1,1,1,1,1,1,1
    fx=0/0/0,2/1/0,4/2/0,6/3/0,8/4/0,10/5/0

Exit status 0 iff every named expectation matches.  Prints one line per
mismatch.  Invariant violations found by mnprog.problems() are always
reported, whatever the expectations are.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mnprog  # noqa: E402


def parse_expect(argv):
    exp = {}
    for tok in argv:
        k, _, v = tok.partition("=")
        if k not in ("size", "name", "nargs", "nfixup", "L", "blk", "tail",
                     "tags", "fx"):
            sys.exit("check_prog: unknown key %r" % k)
        exp[k] = v
    return exp


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path = sys.argv[1]
    exp = parse_expect(sys.argv[2:])
    p = mnprog.parse(path)
    if p is None:
        print("FAIL %s: not a mncore::Program" % path)
        return 1
    bad = 0
    want_name = exp.get("name")
    if want_name is not None and p["name"] != want_name:
        print("FAIL name %r != %r" % (p["name"], want_name))
        bad += 1
    for key, got in (("size", p["size"]), ("nargs", p["nargs"]),
                     ("nfixup", len(p["fixups"])), ("L", p["L"]),
                     ("blk", p["blk"]), ("tail", p["tail"])):
        if key not in exp:
            continue
        want = int(exp[key])
        if got != want:
            print("FAIL %s %d != %d" % (key, got, want))
            bad += 1
    if "tags" in exp:
        want = [int(x) for x in exp["tags"].split(",")]
        if p["tags"] != want:
            print("FAIL tags %s != %s" % (p["tags"], want))
            bad += 1
    if "fx" in exp:
        want = [tuple(int(x) for x in t.split("/")) for t in exp["fx"].split(",")]
        if p["fixups"] != want:
            print("FAIL fx %s != %s" % (p["fixups"], want))
            bad += 1
    errs = mnprog.problems(p)
    if errs:
        print("FAIL invariants: " + "; ".join(errs))
        bad += len(errs)
    if not bad:
        print("PASS %s" % os.path.basename(path))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
