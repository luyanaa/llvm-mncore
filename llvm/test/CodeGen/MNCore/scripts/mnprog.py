#!/usr/bin/env python3
"""Decode a mncore::Program (the only device-loadable MN-Core container).

Schema, recovered from the vendor's own symbols in libmncl.so
(`mncore::CreateProgram` / `mncore::CreateKernel`) and confirmed on 20
mnclc-built Programs:

    table Program  { target:  Target;   kernels: [Kernel]; }
    table Kernel   { name: string;                        // f0
                     arguments_type: [Argument];          // f1  (1-byte tags)
                     arguments: [void (union payload)];   // f2
                     fixups: [Fixup];                     // f3  (struct, 24 B)
                     instblock: [ubyte];                  // f4  (the image)
                     strings: [string]; }                 // f5  (unused, empty)

    enum  Argument : ubyte { DRAM = 1, Scalar = 2 };
    struct Fixup   { bin_index: u64; addr_table_index: u64;
                     storage_spec_index: u64 };

    scalar payload table { f0: [StorageSpec] }  with
    struct StorageSpec { kind: u64 (=9); offset: u64 (=0); size: u64 (2|4|8) };
    the DRAM-pointer payload table has no fields at all.

It is a **64-bit** FlatBuffer: uoffset_t/soffset_t = 8 bytes, voffset_t = 2,
vector lengths = u64, struct stride 8-aligned.  The file is laid out

    [ 0,          L )   FlatBuffer prefix: root table, kernel table, fixups
    [ L,          L+blk )   instblock vector data (the instruction image)
    [ L+blk,      EOF )   earliest-built objects: argument vectors + payloads

with L = (offset of the instblock length u64) + 8.  The image sits *inside*
the FlatBuffer, so the old "header size" is just L.

Usage: mnprog.py [--check] FILE...
"""
import struct
import sys

D = b""
TAGS = {1: "DRAM", 2: "Scalar"}


def u16(o):
    return struct.unpack_from("<H", D, o)[0]


def u64(o):
    return struct.unpack_from("<Q", D, o)[0]


def i64(o):
    return struct.unpack_from("<q", D, o)[0]


def ok(o, n=1):
    return 0 <= o and o + n <= len(D)


def table_at(t):
    """64-bit table: int64 soffset, then u64 fields; voffset 2 bytes each."""
    if not ok(t, 8):
        return None
    soff = i64(t)
    vt = t - soff
    if not ok(vt, 4) or abs(soff) > 4096:
        return None
    vs, ts = u16(vt), u16(vt + 2)
    if not (4 <= vs <= 512 and 8 <= ts <= 8192 and vs % 2 == 0):
        return None
    if not ok(vt, vs):
        return None
    offs = [u16(vt + 4 + 2 * i) for i in range((vs - 4) // 2)]
    if any(o and o + 8 > ts for o in offs):
        return None
    return vt, vs, ts, offs


def fields(t):
    tb = table_at(t)
    return [] if not tb else [(i, t + o) for i, o in enumerate(tb[3]) if o]


def string_at(t):
    if not ok(t, 8):
        return None
    n = u64(t)
    if not (0 < n <= 256) or not ok(t + 8, n):
        return None
    raw = D[t + 8:t + 8 + n]
    if all(32 <= c < 127 for c in raw):
        return raw.decode("ascii")
    return None


def vec_len(t, width=1):
    if not ok(t, 8):
        return None
    n = u64(t)
    if n > 1 << 24 or not ok(t + 8, n * width):
        return None
    return n


def vec_count(t, width):
    """Element count of a vector that must fit inside the file, else None.

    parse() uses this for the two vectors it would otherwise allocate for.
    A non-Program input (a truncated dump, a text file a failed compile left
    behind) puts garbage in the length word; trusting it means a huge range()
    and an OOM instead of a clean "not a Program".
    """
    if not ok(t, 8):
        return None
    return u64(t) if ok(t + 8, u64(t) * width) else None


def root_kernel_table():
    """Program root -> kernels vector -> first Kernel table."""
    root = u64(0)
    for _, p in fields(root):
        kvec = p + u64(p)
        n = vec_len(kvec, 8)
        if not n:
            continue
        ktab = kvec + 8 + u64(kvec + 8)
        if table_at(ktab):
            return root, kvec, ktab
    return root, None, None


def decode_payload(t):
    """Classify one union payload object."""
    if not ok(t, 8):
        return "bad", "offset %d out of range" % t
    tb = table_at(t)
    if not tb:
        return "raw", "not a table"
    vt, vs, ts, offs = tb
    present = [(i, o) for i, o in enumerate(offs) if o]
    if not present:
        return "empty", "vt{%d,%d} no fields (pointer arg)" % (vs, ts)
    out = []
    for i, o in present:
        p = t + o
        v = u64(p)
        tgt = p + v
        n = vec_len(tgt, 24)
        if n is not None:
            specs = [(u64(tgt + 8 + 24 * j), u64(tgt + 16 + 24 * j),
                      u64(tgt + 24 + 24 * j)) for j in range(min(n, 4))]
            out.append("f%d=[StorageSpec{%s}]" % (
                i, " ".join("kind=%d off=%d size=%d" % s for s in specs)))
        else:
            out.append("f%d=u64 %d" % (i, v))
    return "table", "vt{%d,%d} " % (vs, ts) + " ".join(out)


def parse(path):
    """Return the decoded Program as a dict, or None if it is not one."""
    global D
    D = open(path, "rb").read()
    n = len(D)
    root, kvec, ktab = root_kernel_table()
    if ktab is None:
        return None
    f = dict(fields(ktab))
    rf = dict(fields(root))
    p = {
        "path": path,
        "size": n,
        "target": u64(rf[0]) if 0 in rf else 0,
        "target_in_file": 0 in rf,
        "kernels": vec_len(kvec, 8),
        "name": string_at(f[0] + u64(f[0])) if 0 in f else None,
        "tags": [],
        "payloads": [],
        "fixups": [],
        "n_strings": None,
    }
    # instblock: field f4 -> Vector<ubyte>; data starts at L, length at L-8.
    ipos = f[4]
    itgt = ipos + u64(ipos)
    if not ok(itgt, 8):
        return None
    p["blk"] = u64(itgt)
    p["L"] = itgt + 8
    if not ok(p["L"], p["blk"]):
        return None
    p["tail"] = n - p["L"] - p["blk"]
    # arguments_type (f1): 1 byte per argument.
    apos = f[1]
    atgt = apos + u64(apos)
    p["nargs"] = vec_len(atgt, 1) or 0
    p["tags"] = list(D[atgt + 8:atgt + 8 + p["nargs"]])
    # arguments (f2): union vector, one 8-byte offset per argument.
    ppos = f[2]
    ptgt = ppos + u64(ppos)
    npay = vec_count(ptgt, 8)
    if npay is None:
        return None
    for j in range(npay):
        e = ptgt + 8 + 8 * j
        if not ok(e, 8):
            p["payloads"].append(("bad", "element %d out of range" % j))
            continue
        p["payloads"].append(decode_payload(e + u64(e)))
    # fixups (f3): Vector<Fixup>, 24-byte structs.
    fxpos = f[3]
    fxtgt = fxpos + u64(fxpos)
    nfx = vec_count(fxtgt, 24)
    if nfx is None:
        return None
    p["fixups"] = [(u64(fxtgt + 8 + 24 * j), u64(fxtgt + 16 + 24 * j),
                    u64(fxtgt + 24 + 24 * j)) for j in range(nfx)]
    # strings (f5): always an empty vector in practice.
    if 5 in f:
        s = f[5] + u64(f[5])
        p["n_strings"] = u64(s) if ok(s, 8) else None
    return p


def problems(p):
    """Invariants of the schema, as a list of human-readable violations."""
    errs = []
    if p["L"] + p["blk"] + p["tail"] != p["size"]:
        errs.append("L+blk+tail != size")
    if p["blk"] % 128:
        errs.append("instblock not a multiple of 128")
    if p["L"] % 8:
        errs.append("image not 8-aligned")
    if p["tail"] % 8:
        errs.append("tail not 8-aligned")
    if p["nargs"] != len(p["payloads"]):
        errs.append("arguments_type(%d) != payloads(%d)"
                    % (p["nargs"], len(p["payloads"])))
    for j, t in enumerate(p["tags"]):
        if t not in TAGS:
            errs.append("arg[%d] unknown tag %d" % (j, t))
    for j, (kind, txt) in enumerate(p["payloads"]):
        if j >= len(p["tags"]):
            break
        if p["tags"][j] == 2 and "StorageSpec" not in txt:
            errs.append("scalar payload[%d] has no storage spec: %s" % (j, txt))
        if p["tags"][j] == 1 and kind != "empty":
            errs.append("pointer payload[%d] is not empty: %s" % (j, txt))
    for j, (a, b, c) in enumerate(p["fixups"]):
        if b >= p["nargs"]:
            errs.append("fixup[%d] addr_table_index %d >= n_args" % (j, b))
        if c * 128 > p["blk"] * 8:
            errs.append("fixup[%d] storage_spec_index %d implausible" % (j, c))
        if j and a <= p["fixups"][j - 1][0]:
            errs.append("fixup[%d] bin_index not increasing" % j)
    return errs


def show(p):
    print("=== %s  size %d" % (p["path"], p["size"]))
    print("  Program  target=%d%s  kernels=%d" %
          (p["target"],
           "" if p["target_in_file"] else " (field absent = default)",
           p["kernels"]))
    print("  Kernel   name=%r  n_args=%d  n_fixup=%d  n_strings=%s" %
          (p["name"], p["nargs"], len(p["fixups"]), p["n_strings"]))
    print("  image    L=%d  blk=%d  tail=%d  (file = L+blk+tail)" %
          (p["L"], p["blk"], p["tail"]))
    print("  args     %s" % ", ".join("[%d]%s" % (i, TAGS.get(t, "?%d" % t))
                                     for i, t in enumerate(p["tags"])))
    for j, (kind, txt) in enumerate(p["payloads"]):
        print("  payload[%d] %-5s %s" % (j, kind, txt))
    for j, fx in enumerate(p["fixups"][:12]):
        print("  fixup[%d] bin_index=%d addr_table_index=%d storage_spec_index=%d"
              % ((j,) + fx))
    if len(p["fixups"]) > 12:
        print("  fixup[12..%d] ..." % (len(p["fixups"]) - 1))


def decode(path, check=False):
    p = parse(path)
    if p is None:
        print("%s: not a mncore::Program" % path)
        return False
    show(p)
    if not check:
        return True
    errs = problems(p)
    print("  CHECK    %s" % ("OK" if not errs else "FAIL: " + "; ".join(errs)))
    return not errs


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    bad = [p for p in args if not decode(p, check)]
    if check:
        print("\n%d/%d checked OK" % (len(args) - len(bad), len(args)))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
