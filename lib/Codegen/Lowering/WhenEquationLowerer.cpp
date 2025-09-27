#include "marco/Codegen/Lowering/BaseModelica/WhenEquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
WhenEquationLowerer::WhenEquationLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool WhenEquationLowerer::lower(const ast::bmodelica::WhenEquation &equation) {
  llvm_unreachable("When equation is not implemented yet");
  return false;
}
} // namespace marco::codegen::lowering::bmodelica
