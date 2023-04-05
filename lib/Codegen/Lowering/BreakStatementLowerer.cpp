#include "marco/Codegen/Lowering/BreakStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  BreakStatementLowerer::BreakStatementLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void BreakStatementLowerer::lower(const ast::BreakStatement& statement)
  {
    mlir::Location location = loc(statement.getLocation());
    builder().create<BreakOp>(location);
  }
}
