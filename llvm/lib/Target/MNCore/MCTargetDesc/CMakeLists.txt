add_llvm_component_library(LLVMMNCoreDesc
  MNCoreInstPrinter.cpp
  MNCoreMCAsmInfo.cpp
  MNCoreMCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  Support
  MNCoreInfo

  ADD_TO_COMPONENT
  MNCore
  )
