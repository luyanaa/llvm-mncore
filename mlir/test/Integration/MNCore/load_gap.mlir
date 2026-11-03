// load -> independent ALU -> consumer. The independent expression must remain
// schedulable; only the load-to-consumer pair is separated by the packetizer
// + hazard pass (no explicit fence).

func.func @kernel(%b: vector<4xf32>, %c: vector<4xf32>) {
  %lb = "mncore.lbop"() <{mem = 0 : i32, addr = 0 : i64, width = 2 : i32, flags = 0 : i32}> : () -> !mncore.token
  %hn = "mncore.none"() : () -> !mncore.token
  %a = "mncore.vec.load"(%lb, %hn) <{mode = 0 : i64, zmask = -1 : i64, wmask = -1 : i64}> : (!mncore.token, !mncore.token) -> vector<4xf32>
  %x = "mncore.vec.add"(%b, %c) <{zmask = -1 : i64, wmask = -1 : i64}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %r = "mncore.vec.fma"(%a, %x, %c) <{zmask = -1 : i64, wmask = -1 : i64}> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  "mncore.mat.write"(%r) <{face = 0 : i64, row = 0 : i64}> : (vector<4xf32>) -> ()
  return
}
