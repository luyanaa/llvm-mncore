add_llvm_component_group(MNCore)

set(LLVM_TARGET_DEFINITIONS MNCore.td)

tablegen(LLVM MNCoreGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM MNCoreGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM MNCoreGenDFAPacketizer.inc -gen-dfa-packetizer)
tablegen(LLVM MNCoreGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM MNCoreGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM MNCoreGenSDNodeInfo.inc -gen-sd-node-info)
tablegen(LLVM MNCoreGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(MNCoreCommonTableGen)

add_llvm_target(MNCoreCodeGen
  MNCoreAsmPrinter.cpp
  MNCoreFrameLowering.cpp
  MNCoreInstrInfo.cpp
  MNCoreISelDAGToDAG.cpp
  MNCoreISelLowering.cpp
  MNCoreMCInstLower.cpp
  MNCorePacketizer.cpp
  MNCoreRegisterInfo.cpp
  MNCoreSelectionDAGInfo.cpp
  MNCoreSubtarget.cpp
  MNCoreTargetMachine.cpp

  LINK_COMPONENTS
  MNCoreDesc
  MNCoreInfo
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  SelectionDAG
  Support
  Target
  TargetParser

  ADD_TO_COMPONENT
  MNCore
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
