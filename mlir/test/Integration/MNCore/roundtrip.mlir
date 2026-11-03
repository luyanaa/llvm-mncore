// T5 numeric round-trip at the instruction-contract level: distributed
// two-plane load -> elementwise add -> distributed two-plane collect.

func.func @kernel() {
  %lo = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 1 : i32, flags = 0 : i32}> : () -> !mncore.token
  %hi = "mncore.lbop"() <{mem = 0 : i32, addr = 64 : i64, width = 1 : i32, flags = 0 : i32}> : () -> !mncore.token
  %a = "mncore.vec.load"(%lo, %hi) <{mode = 3 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token, !mncore.token) -> vector<4xf32>
  %r = "mncore.vec.add"(%a, %a) <{zmask = -1 : i64, wmask = -1 : i64}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %dlo = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 1 : i32, flags = 0 : i32}> : () -> !mncore.token
  %dhi = "mncore.lbop"() <{mem = 0 : i32, addr = 64 : i64, width = 1 : i32, flags = 0 : i32}> : () -> !mncore.token
  "mncore.vec.store"(%r, %dlo, %dhi) <{mode = 3 : i64, set = 0 : i64}> : (vector<4xf32>, !mncore.token, !mncore.token) -> ()
  return
}
