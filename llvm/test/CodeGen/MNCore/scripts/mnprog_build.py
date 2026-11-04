#!/usr/bin/env python3
"""Rebuild a mncore::Program from its decoded fields, byte for byte.

Companion to mnprog.py.  It is deliberately a *from-scratch* 64-bit
FlatBuffers writer (uoffset/soffset = 8, voffset = 2, vector length = u64)
rather than a wrapper around a library, because the point is to prove the
recovered schema is complete: if the rebuilt file is identical to the one
mnclc produced, no header field is left unexplained.

Construction order and the quirks that matter are documented in
docs/mncore_mnir_offload.md §6 A.1:

  * payload tables, arguments_type, arguments, instblock, fixups, strings,
    then Kernel -> kernels -> root;
  * the instblock vector is created with 64-byte alignment;
  * identical vtables are shared (soffset goes negative when the table sits
    before the vtable it reuses).

Usage: mnprog_build.py [--write OUT] FILE...
       mnprog_build.py --check FILE...     # rebuild and compare in memory
"""
import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import mnprog  # noqa: E402

U64 = 8


class Builder64:
    """Minimal backwards FlatBuffers builder with 8-byte offsets."""

    def __init__(self):
        self.b = bytearray()      # file-order prefix; new objects are prepended
        self.minalign = 1
        self.depth = 0
        self.vtables = {}         # vtable bytes -> position (bytes written)
        self.slots = None
        self.obj_end = 0

    # -- primitives ---------------------------------------------------
    def pos(self):
        return len(self.b)

    def pad(self, n):
        if n:
            self.b[0:0] = b"\0" * n

    def prep(self, size, additional):
        if size > self.minalign:
            self.minalign = size
        self.pad((-(self.pos() + additional)) % size)

    def place(self, value, width=U64):
        self.prep(width, 0)
        self.b[0:0] = value.to_bytes(width, "little")

    def place_uoffset(self, target):
        """Store a forward offset from this field to `target`."""
        self.prep(U64, 0)
        self.b[0:0] = (self.pos() + U64 - target).to_bytes(U64, "little")

    # -- vectors ------------------------------------------------------
    def start_vector(self, elem_size, count, align):
        self.depth += 1
        self.prep(U64, elem_size * count)
        self.prep(align, elem_size * count)

    def end_vector(self, count):
        self.depth -= 1
        self.place(count, U64)
        return self.pos()

    def vector_of_scalars(self, items, elem_size, align):
        """items are bytes objects already in struct/big-endian-free form."""
        self.start_vector(elem_size, len(items), align)
        for it in reversed(items):
            self.pad((-self.pos()) % align)
            self.b[0:0] = it
        return self.end_vector(len(items))

    def vector_of_bytes(self, data, align=1):
        self.start_vector(1, len(data), align)
        self.b[0:0] = data
        return self.end_vector(len(data))

    def vector_of_offsets(self, offsets):
        self.start_vector(8, len(offsets), 8)
        for off in reversed(offsets):
            self.place_uoffset(off)
        return self.end_vector(len(offsets))

    # -- tables -------------------------------------------------------
    def start_object(self, nfields):
        self.depth += 1
        self.slots = [0] * nfields
        self.obj_end = self.pos()

    def slot_scalar(self, slot, value, default, width=U64):
        if value != default:
            self.place(value, width)
            self.slots[slot] = self.pos()

    def slot_offset(self, slot, off):
        if off:
            self.place_uoffset(off)
            self.slots[slot] = self.pos()

    def end_object(self):
        self.depth -= 1
        self.place(0, U64)                      # soffset placeholder
        table = self.pos()
        table_size = table - self.obj_end
        entries = [table - s if s else 0 for s in self.slots]
        while entries and entries[-1] == 0:     # flatbuffers trims the tail
            entries.pop()
        vt = struct.pack("<HH", 4 + 2 * len(entries), table_size)
        vt += b"".join(struct.pack("<H", e) for e in entries)
        if vt in self.vtables:
            vpos = self.vtables[vt]
        else:
            self.b[0:0] = vt
            vpos = self.pos()
            self.vtables[vt] = vpos
        idx = self.pos() - table          # bytes prepended since the soffset
        self.b[idx:idx + U64] = struct.pack("<q", vpos - table)
        return table


def build(p):
    """Rebuild the Program described by mnprog.parse()'s dict."""
    b = Builder64()
    img = p["image"]
    tags = p["tags"]
    specs = p["specs"]

    payloads = []
    for i, tag in enumerate(tags):
        if tag == 2:
            vec = b.vector_of_scalars([struct.pack("<QQQ", *specs[i])], 24, 8)
            b.start_object(1)
            b.slot_offset(0, vec)
            payloads.append(b.end_object())
        else:
            b.start_object(0)
            payloads.append(b.end_object())

    args_type = b.vector_of_scalars([bytes([t]) for t in tags], 1, 1)
    args_union = b.vector_of_offsets(payloads)
    instblock = b.vector_of_bytes(img, align=64)
    fixups = b.vector_of_scalars(
        [struct.pack("<QQQ", *f) for f in p["fixups"]], 24, 8)
    strings = b.vector_of_scalars([], 8, 8)
    name_off = make_string(b, p["name"] or "k")

    b.start_object(6)
    b.slot_offset(5, strings)
    b.slot_offset(4, instblock)
    b.slot_offset(3, fixups)
    b.slot_offset(2, args_union)
    b.slot_offset(1, args_type)
    b.slot_offset(0, name_off)
    kernel = b.end_object()

    kernels = b.vector_of_offsets([kernel])
    b.start_object(2)
    b.slot_offset(1, kernels)
    b.slot_scalar(0, p["target"], 0)
    root = b.end_object()
    b.prep(b.minalign, U64)
    b.place_uoffset(root)
    return bytes(b.b)


def make_string(b, s):
    raw = s.encode()
    b.start_vector(1, len(raw) + 1, 1)
    b.b[0:0] = b"\0"
    b.b[0:0] = raw
    return b.end_vector(len(raw))


def spec_list(D, p):
    """Recover the StorageSpec triplet of each scalar argument."""
    mnprog.D = D
    root, kvec, ktab = mnprog.root_kernel_table()
    f = dict(mnprog.fields(ktab))
    t = f[2] + mnprog.u64(f[2])
    out = [None] * p["nargs"]
    for j in range(min(mnprog.u64(t), len(out))):
        if p["tags"][j] != 2:
            continue
        tbl = t + 8 + 8 * j
        tbl += mnprog.u64(tbl)
        fo = mnprog.table_at(tbl)[3][0]
        vec = tbl + fo + mnprog.u64(tbl + fo)
        out[j] = (mnprog.u64(vec + 8), mnprog.u64(vec + 16),
                  mnprog.u64(vec + 24))
    return out


def rebuild(path):
    p = mnprog.parse(path)
    if p is None:
        return None, None
    p["image"] = mnprog.D[p["L"]:p["L"] + p["blk"]]
    p["specs"] = spec_list(mnprog.D, p)
    return build(p), mnprog.D


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = None
    if "--write" in sys.argv:
        out = sys.argv[sys.argv.index("--write") + 1]
        args = [a for a in args if a != out]
    bad = 0
    for path in args:
        got, want = rebuild(path)
        if got is None:
            print("%s: not a Program" % path)
            bad += 1
            continue
        if out:
            open(out, "wb").write(got)
        if got == want:
            print("IDENTICAL %s (%d B)" % (path, len(want)))
            continue
        diff = next((i for i in range(min(len(got), len(want)))
                     if got[i] != want[i]), min(len(got), len(want)))
        print("DIFFERS   %s: len %d vs %d, first difference at %d"
              % (path, len(got), len(want), diff))
        bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
