add_mlir_conversion_library(MLIRMNCoreToLLVM
  MNCoreToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MNCoreToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRIR
  MLIRLLVMDialect
  MLIRMNCoreDialect
  MLIRPass
  MLIRTransforms
  )
