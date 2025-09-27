#include "marco/Codegen/Lowering/BaseModelica/IfEquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
IfEquationLowerer::IfEquationLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool IfEquationLowerer::lower(const ast::bmodelica::IfEquation &equation) {
  llvm_unreachable("If equation is not implemented yet");
  return false;
}
} // namespace marco::codegen::lowering::bmodelica
