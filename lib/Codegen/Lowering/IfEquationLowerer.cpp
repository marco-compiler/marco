#include "marco/Codegen/Lowering/BaseModelica/IfEquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
IfEquationLowerer::IfEquationLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool IfEquationLowerer::lower(const ast::bmodelica::IfEquation &equation) {
  ScopedSymbolTable::Scope scope(getScopedSymbolTable());
  mlir::Location location = loc(equation.getLocation());
  size_t numberOfEquations = equation.getNumOfIfEquations();

  IfEquationOp ifEquationOp = builder().create<IfEquationOp>(location);

  builder().setInsertionPointToStart(ifEquationOp.conditionBlock());
  std::optional<Results> loweredCondition = lower(*equation.getIfCondition());
  if (!loweredCondition) {
    return false;
  }
  assert(loweredCondition->size() == 1);
  mlir::Value condition = (*loweredCondition)[0].get(location);
  builder().create<YieldOp>(location, condition);

  builder().setInsertionPointToStart(ifEquationOp.thenBlock());
  for (size_t i = 0, e = equation.getNumOfIfEquations(); i < e; ++i) {
    if (!lower(*equation.getIfEquation(i))) {
      return false;
    }
  }

  assert(equation.getNumOfElseIfConditions() == 0 &&
         "else-if chains are not supported");

  assert(equation.getNumOfElseEquations() == numberOfEquations &&
         "Number of equaitons of else clause must match number of equations in "
         "the if clause.");

  builder().setInsertionPointToStart(ifEquationOp.elseBlock());
  for (size_t i = 0, e = equation.getNumOfElseEquations(); i < e; ++i) {
    if (!lower(*equation.getElseEquation(i))) {
      return false;
    }
  }

  builder().setInsertionPointAfter(ifEquationOp);

  return true;
}
} // namespace marco::codegen::lowering::bmodelica
