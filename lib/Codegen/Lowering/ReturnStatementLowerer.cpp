#include "marco/Codegen/Lowering/ReturnStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
ReturnStatementLowerer::ReturnStatementLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool ReturnStatementLowerer::lower(const ast::ReturnStatement &statement) {
  mlir::Location location = loc(statement.getLocation());
  builder().create<ReturnOp>(location);
  return true;
}
} // namespace marco::codegen::lowering
