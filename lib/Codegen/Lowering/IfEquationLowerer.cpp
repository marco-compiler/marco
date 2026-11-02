#include "marco/Codegen/Lowering/BaseModelica/IfEquationLowerer.h"
#include "marco/AST/BaseModelica/Constant.h"
#include "marco/AST/BaseModelica/Equation.h"
#include "marco/AST/BaseModelica/Expression.h"
#include "marco/Dialect/BaseModelica/IR/Ops.h"
#include <cassert>
#include <functional>

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

  // OMC lowers the `else` branch into an `elseif true then` branch, so the else
  // equations may arrive in the (single) else-if bucket with a `true`
  // condition. Select the source of the else equations accordingly.
  std::function<const ast::bmodelica::Equation *(size_t)> getElseEquation;
  if (equation.getNumOfElseEquations() != numberOfEquations) {
    assert(equation.getNumOfElseIfConditions() == 1 &&
           equation.getNumOfElseIfEquations(0) == numberOfEquations &&
           "Number of equations of else clause must match number of equations "
           "in the if clause.");

    const ast::bmodelica::Expression *elseIfCondition =
        equation.getElseIfCondition(0);
    assert(elseIfCondition->isa<ast::bmodelica::Constant>() &&
           elseIfCondition->cast<ast::bmodelica::Constant>()->as<bool>() &&
           "else-if chains are not supported in if-equations; the only "
           "accepted else-if condition is the literal 'true' introduced by the "
           "frontend for a plain else branch.");

    getElseEquation =
        [&equation](size_t equationNumber) -> const ast::bmodelica::Equation * {
      return equation.getElseIfEquation(0, equationNumber);
    };
  } else {
    assert(equation.getNumOfElseIfConditions() == 0 &&
           "else-if statements are not supported in if-equations.");
    getElseEquation =
        [&equation](size_t equationNumber) -> const ast::bmodelica::Equation * {
      return equation.getElseEquation(equationNumber);
    };
  }

  builder().setInsertionPointToStart(ifEquationOp.elseBlock());
  for (size_t i = 0; i < numberOfEquations; ++i) {
    if (!lower(*getElseEquation(i))) {
      return false;
    }
  }

  builder().setInsertionPointAfter(ifEquationOp);

  return true;
}
} // namespace marco::codegen::lowering::bmodelica
