#include "marco/Codegen/Lowering/AlgorithmLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  AlgorithmLowerer::AlgorithmLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void AlgorithmLowerer::lower(const ast::Algorithm& algorithm)
  {
    mlir::Location location = loc(algorithm.getLocation());

    auto algorithmOp = builder().create<AlgorithmOp>(location);
    assert(algorithmOp.getBodyRegion().empty());

    mlir::Block* algorithmBody =
        builder().createBlock(&algorithmOp.getBodyRegion());

    builder().setInsertionPointToStart(algorithmBody);

    for (size_t i = 0, e = algorithm.size(); i < e; ++i) {
      lower(*algorithm[i]);
    }
  }
}
