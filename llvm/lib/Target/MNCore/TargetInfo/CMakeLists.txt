add_llvm_component_library(LLVMMNCoreInfo
  MNCoreTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  MNCore
  )
