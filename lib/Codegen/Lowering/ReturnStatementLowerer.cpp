#include "marco/Codegen/Lowering/ReturnStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ReturnStatementLowerer::ReturnStatementLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void ReturnStatementLowerer::lower(const ast::ReturnStatement& statement)
  {
    mlir::Location location = loc(statement.getLocation());
    builder().create<ReturnOp>(location);
  }
}
