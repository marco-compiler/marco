add_mlir_library(MLIRCustomTransforms
  DataRecomputation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRCustomTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRFunctionInterfaces
  MLIRLoopLikeInterface
  MLIRMemorySlotInterfaces
  MLIRPass
  MLIRRuntimeVerifiableOpInterface
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransformUtils
  )
