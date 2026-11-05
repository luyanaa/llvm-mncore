// Straight-line MN-Core kernel used by e2e.sh: an L1BM load feeds two typed
// FMAs, which then drive a matrix expression. Typed values must not escape
// (the Layer B pass rejects it). The f32 matrix roles are 4 longwords, so the
// 2-longword FMA result enters through the matvec x, not as a slab.

func.func @kernel(%b: vector<4xf32>, %c: vector<4xf32>) {
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %hn = "mncore.none"() : () -> !mncore.token
  %a = "mncore.vec.load"(%lb, %hn) <{mode = 0 : i64, zmask = -1 : i64, wmask = -1 : i64}> :
      (!mncore.token, !mncore.token) -> vector<4xf32>
  %r = "mncore.vec.fma"(%a, %b, %c) <{zmask = -1 : i64, wmask = -1 : i64}> :
      (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %s = "mncore.vec.fma"(%r, %b, %c) <{zmask = -1 : i64, wmask = -1 : i64}> :
      (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %y = "mncore.mat.read"() <{face = 1 : i64, row = 0 : i64}> : () -> vector<8xf32>
  %m = "mncore.mat.vec"(%s, %y) <{face = 0 : i64}> :
      (vector<4xf32>, vector<8xf32>) -> vector<8xf32>
  "mncore.mat.write"(%m) <{face = 0 : i64, row = 0 : i64}> : (vector<8xf32>) -> ()
  return
}
