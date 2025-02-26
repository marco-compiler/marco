#include "marco/Codegen/Lowering/EquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
EquationLowerer::EquationLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

bool EquationLowerer::lower(const ast::Equation &equation) {
  if (auto casted = equation.dyn_cast<ast::EqualityEquation>()) {
    return lower(*casted);
  }

  if (auto casted = equation.dyn_cast<ast::IfEquation>()) {
    return lower(*casted);
  }

  if (auto casted = equation.dyn_cast<ast::ForEquation>()) {
    return lower(*casted);
  }

  if (auto casted = equation.dyn_cast<ast::WhenEquation>()) {
    return lower(*casted);
  }

  llvm_unreachable("Unknown equation type");
  return false;
}
} // namespace marco::codegen::lowering
