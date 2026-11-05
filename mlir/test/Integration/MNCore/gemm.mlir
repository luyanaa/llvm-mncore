// GEMM tile end-to-end: one 8x4x4 block lowers to two matrix-slab writes and
// one matvec, i.e. exactly the MAU expression pair the vendor's own GEMM
// emits (design doc 22.7.5). This is a STRUCTURAL golden: it pins instruction
// selection, not a numeric result. Neither gpfn3 (dst := y, 22.9) nor emu2
// (mncl only emits the $mauf form the emulator cannot feed, 22.12) can
// validate a product today, so `tl.dot` remains fail-closed (22.5).

func.func @kernel(%alo: vector<8xf32>, %ahi: vector<8xf32>,
                  %b: vector<4xf32>, %c0: vector<8xf32>) {
  %c = "mncore.gemm.tile"(%alo, %ahi, %b, %c0) <{face = 0 : i64}> :
      (vector<8xf32>, vector<8xf32>, vector<4xf32>, vector<8xf32>) -> vector<8xf32>
  "mncore.mat.write"(%c) <{face = 1 : i64, row = 0 : i64}> : (vector<8xf32>) -> ()
  return
}
