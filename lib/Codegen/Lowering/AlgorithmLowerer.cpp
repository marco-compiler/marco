#include "marco/Codegen/Lowering/AlgorithmLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
AlgorithmLowerer::AlgorithmLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

bool AlgorithmLowerer::lower(const ast::Algorithm &algorithm) {
  if (algorithm.empty()) {
    return true;
  }

  mlir::Location location = loc(algorithm.getLocation());

  auto algorithmOp = builder().create<AlgorithmOp>(location);
  assert(algorithmOp.getBodyRegion().empty());

  mlir::Block *algorithmBody =
      builder().createBlock(&algorithmOp.getBodyRegion());

  builder().setInsertionPointToStart(algorithmBody);

  for (size_t i = 0, e = algorithm.size(); i < e; ++i) {
    if (!lower(*algorithm[i])) {
      return false;
    }
  }

  builder().setInsertionPointAfter(algorithmOp);
  return true;
}
} // namespace marco::codegen::lowering
