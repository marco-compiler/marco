#include "marco/Codegen/Lowering/BaseModelica/EquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
EquationLowerer::EquationLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

bool EquationLowerer::lower(const ast::bmodelica::Equation &equation) {
  if (auto casted = equation.dyn_cast<ast::bmodelica::EqualityEquation>()) {
    return lower(*casted);
  }

  if (auto casted = equation.dyn_cast<ast::bmodelica::IfEquation>()) {
    return lower(*casted);
  }

  if (auto casted = equation.dyn_cast<ast::bmodelica::ForEquation>()) {
    return lower(*casted);
  }

  if (auto casted = equation.dyn_cast<ast::bmodelica::WhenEquation>()) {
    return lower(*casted);
  }

  llvm_unreachable("Unknown equation type");
  return false;
}
} // namespace marco::codegen::lowering::bmodelica
