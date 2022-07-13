#include "marco/Codegen/Lowering/AlgorithmLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  AlgorithmLowerer::AlgorithmLowerer(LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge)
  {
  }

  void AlgorithmLowerer::lower(const ast::Algorithm& algorithm)
  {
    auto algorithmOp = builder().create<AlgorithmOp>(loc(algorithm.getLocation()));
    assert(algorithmOp.getBodyRegion().empty());
    mlir::Block* algorithmBody = builder().createBlock(&algorithmOp.getBodyRegion());
    builder().setInsertionPointToStart(algorithmBody);

    for (const auto& statement : algorithm.getBody()) {
      lower(*statement);
    }
  }
}
