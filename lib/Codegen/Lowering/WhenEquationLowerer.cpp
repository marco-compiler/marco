#include "marco/Codegen/Lowering/WhenEquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
WhenEquationLowerer::WhenEquationLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool WhenEquationLowerer::lower(const ast::WhenEquation &equation) {
  llvm_unreachable("When equation is not implemented yet");
  return false;
}
} // namespace marco::codegen::lowering
