# Program-container fixtures

Four `mncore::Program` files produced by the vendor `mnclc`
(`sde64 -hsw -- mnclc --allow_pe_args --mnir --dump -e k -o <out>.bin <src>.c`,
mncl build 2026-10-15), kept here so the recovered FlatBuffers schema
(`docs/mncore_mnir_offload.md` §6 A.1) has a local, toolchain-free regression.
`scripts/check_prog.py` decodes each and pins the fields named in
`scripts/run_regression.sh`; `scripts/mnprog.py` is the decoder itself.

| file | fixups | args | blk | L | tail | what it pins |
|---|---|---|---|---|---|---|
| `ke.bin` | 0 | 0 | 0 | 192 | 64 | empty image, no arguments — the smallest valid Program |
| `s28.bin` | 1 | 2 | 256 | 192 | 64 | `tail == 64` with a fixup vector present |
| `gen26.bin` | 3 | 2 | 768 | 256 | 64 | `L = 256` bucket; fixup `addr_table_index` distinguishes the two args |
| `gen16.bin` | 6 | 7 | 1536 | 320 | 192 | `tail = 192` with 7 pointer args; once mis-read as "another layout" |

Do not regenerate these in place: the pinned numbers are the evidence. If a
new mnclc build changes them, that is a finding, not a fixture update.
