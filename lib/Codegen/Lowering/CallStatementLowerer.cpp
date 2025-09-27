#include "marco/Codegen/Lowering/BaseModelica/CallStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
CallStatementLowerer::CallStatementLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool CallStatementLowerer::lower(
    const ast::bmodelica::CallStatement &statement) {
  return static_cast<bool>(lower(*statement.getCall()));
}
} // namespace marco::codegen::lowering::bmodelica
