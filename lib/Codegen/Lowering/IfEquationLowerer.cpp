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

  std::optional<Results> loweredCondition = lower(*equation.getIfCondition());
  if (!loweredCondition) {
    return false;
  }
  assert(loweredCondition->size() == 1);
  mlir::Value condition = (*loweredCondition)[0].get(location);

  auto ifEquationOp = builder().create<IfEquationOp>(location, condition);

  builder().setInsertionPointToStart(ifEquationOp.thenBlock());
  for (size_t i = 0, e = equation.getNumOfIfEquations(); i < e; ++i) {
    if (!lower(*equation.getIfEquation(i))) {
      return false;
    }
  }

  builder().setInsertionPointToStart(ifEquationOp.elseBlock());
  for (size_t i = 0, e = equation.getNumOfElseIfConditions(); i < e; ++i) {
    assert(equation.getNumOfElseIfEquations(i) == numberOfEquations &&
           "Number of equaitons of if-else clause must match number of "
           "equations in the if clause.");

    auto loweredElseIfCondition = lower(*equation.getElseIfCondition(i));
    if (!loweredElseIfCondition) {
      return false;
    }
    assert(loweredElseIfCondition->size() == 1);
    mlir::Value elseIfCondition = (*loweredElseIfCondition)[0].get(location);

    auto elseIfOp = builder().create<IfEquationOp>(location, elseIfCondition);

    builder().setInsertionPointToStart(elseIfOp.thenBlock());
    for (size_t j = 0, je = equation.getNumOfElseIfEquations(i); j < je; ++j) {
      if (!lower(*equation.getElseIfEquation(i, j))) {
        return false;
      }
    }

    builder().setInsertionPointToStart(elseIfOp.elseBlock());
  }

  assert(equation.getNumOfElseEquations() == numberOfEquations &&
         "Number of equaitons of else clause must match number of equations in "
         "the if clause.");
  for (size_t i = 0, e = equation.getNumOfElseEquations(); i < e; ++i) {
    if (!lower(*equation.getElseEquation(i))) {
      return false;
    }
  }

  builder().setInsertionPointAfter(ifEquationOp);

  return true;
}
} // namespace marco::codegen::lowering::bmodelica
