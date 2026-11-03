add_mlir_dialect_library(MLIRMNCoreDialect
  MNCoreDialect.cpp
  MNCoreOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MNCore

  DEPENDS
  MLIRMNCoreOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  )
