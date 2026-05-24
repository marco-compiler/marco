#include "marco/Codegen/Lowering/BaseModelica/CallEquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
CallEquationLowerer::CallEquationLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool CallEquationLowerer::lower(const ast::bmodelica::CallEquation &equation) {
  mlir::Location location = loc(equation.getLocation());

  mlir::emitWarning(location,
                    "lowering of call equations is not yet implemented");

  return true;
}
} // namespace marco::codegen::lowering::bmodelica
