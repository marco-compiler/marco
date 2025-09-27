#include "marco/Codegen/Lowering/BaseModelica/BreakStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
BreakStatementLowerer::BreakStatementLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool BreakStatementLowerer::lower(
    const ast::bmodelica::BreakStatement &statement) {
  mlir::Location location = loc(statement.getLocation());
  builder().create<BreakOp>(location);
  return true;
}
} // namespace marco::codegen::lowering::bmodelica
