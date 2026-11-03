// Straight-line MN-Core kernel used by e2e.sh: an L1BM load feeds two typed
// FMAs written into the matrix register. Typed values must not escape (the
// Layer B pass rejects it).

func.func @kernel(%b: vector<4xf32>, %c: vector<4xf32>) {
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %a = "mncore.vec.load"(%lb) <{mode = 0 : i64, zmask = -1 : i64, wmask = -1 : i64}> :
      (!mncore.token) -> vector<4xf32>
  %r = "mncore.vec.fma"(%a, %b, %c) <{zmask = -1 : i64, wmask = -1 : i64}> :
      (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %s = "mncore.vec.fma"(%r, %b, %c) <{zmask = -1 : i64, wmask = -1 : i64}> :
      (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  "mncore.mat.write"(%s) <{face = 0 : i64, row = 0 : i64}> : (vector<4xf32>) -> ()
  return
}
