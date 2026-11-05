// Distributed load: l1bmd has no double-longword form, so a 4xf32/PE vector
// is loaded as two longword planes (low at L1BM 0, high at L1BM 64).

func.func @kernel(%b: vector<4xf32>, %c: vector<4xf32>) {
  %lo = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 1 : i32, flags = 0 : i32}> : () -> !mncore.token
  %hi = "mncore.lbop"() <{mem = 0 : i32, addr = 64 : i64, width = 1 : i32, flags = 0 : i32}> : () -> !mncore.token
  %a = "mncore.vec.load"(%lo, %hi) <{mode = 3 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token, !mncore.token) -> vector<4xf32>
  %r = "mncore.vec.fma"(%a, %b, %c) <{zmask = -1 : i64, wmask = -1 : i64}> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %y = "mncore.mat.read"() <{face = 1 : i64, row = 0 : i64}> : () -> vector<8xf32>
  %m = "mncore.mat.vec"(%r, %y) <{face = 0 : i64}> : (vector<4xf32>, vector<8xf32>) -> vector<8xf32>
  "mncore.mat.write"(%m) <{face = 0 : i64, row = 0 : i64}> : (vector<8xf32>) -> ()
  return
}
