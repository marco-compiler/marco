#include "marco/Codegen/Lowering/BaseModelica/EqualityEquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
EqualityEquationLowerer::EqualityEquationLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool EqualityEquationLowerer::lower(
    const ast::bmodelica::EqualityEquation &equation) {
  mlir::Location location = loc(equation.getLocation());

  auto equationOp = builder().create<EquationOp>(location);
  assert(equationOp.getBodyRegion().empty());

  mlir::Block *bodyBlock = builder().createBlock(&equationOp.getBodyRegion());

  builder().setInsertionPointToStart(bodyBlock);

  llvm::SmallVector<mlir::Value, 1> lhs;
  llvm::SmallVector<mlir::Value, 1> rhs;

  {
    // Left-hand side.
    const auto *expression = equation.getLhsExpression();

    auto referencesLoc = loc(expression->getLocation());
    auto references = lower(*expression);
    if (!references) {
      return false;
    }

    for (auto &reference : *references) {
      lhs.push_back(reference.get(referencesLoc));
    }
  }

  {
    // Right-hand side.
    const auto *expression = equation.getRhsExpression();

    auto referencesLoc = loc(expression->getLocation());
    auto references = lower(*expression);
    if (!references) {
      return false;
    }

    for (auto &reference : *references) {
      rhs.push_back(reference.get(referencesLoc));
    }
  }

  mlir::Value lhsTuple = builder().create<EquationSideOp>(location, lhs);
  mlir::Value rhsTuple = builder().create<EquationSideOp>(location, rhs);
  builder().create<EquationSidesOp>(location, lhsTuple, rhsTuple);
  builder().setInsertionPointAfter(equationOp);

  return true;
}
} // namespace marco::codegen::lowering::bmodelica
