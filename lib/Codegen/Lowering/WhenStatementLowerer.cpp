#include "marco/Codegen/Lowering/BaseModelica/WhenStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
WhenStatementLowerer::WhenStatementLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool WhenStatementLowerer::lower(
    const ast::bmodelica::WhenStatement &statement) {
  llvm_unreachable("When statement is not implemented yet");
  return false;
}
} // namespace marco::codegen::lowering::bmodelica
