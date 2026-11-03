// DRAM -> L2BM -> L1BM copy. The loader must bind src_addr/l1_addr through the
// relocation contract; this kernel only checks the instruction sequence.

func.func @kernel() {
  "mncore.dma"() <{src_space = 1 : i32, src_addr = 0 : i64, l2_addr = 0 : i64,
                    l1_addr = 0 : i64, size = 64 : i64, tag = 1 : i32,
                    mv_flags = 0 : i32}> : () -> ()
  return
}
