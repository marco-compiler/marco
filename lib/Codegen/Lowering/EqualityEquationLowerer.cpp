#include "marco/Codegen/Lowering/EqualityEquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  EqualityEquationLowerer::EqualityEquationLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void EqualityEquationLowerer::lower(
      const ast::EqualityEquation& equation)
  {
    mlir::Location location = loc(equation.getLocation());

    auto equationOp = builder().create<EquationOp>(location);
    assert(equationOp.getBodyRegion().empty());

    mlir::Block* bodyBlock =
        builder().createBlock(&equationOp.getBodyRegion());

    builder().setInsertionPointToStart(bodyBlock);

    llvm::SmallVector<mlir::Value, 1> lhs;
    llvm::SmallVector<mlir::Value, 1> rhs;

    {
      // Left-hand side.
      const auto* expression = equation.getLhsExpression();

      auto referencesLoc = loc(expression->getLocation());
      auto references = lower(*expression);

      for (auto& reference : references) {
        lhs.push_back(reference.get(referencesLoc));
      }
    }

    {
      // Right-hand side.
      const auto* expression = equation.getRhsExpression();

      auto referencesLoc = loc(expression->getLocation());
      auto references = lower(*expression);

      for (auto& reference : references) {
        rhs.push_back(reference.get(referencesLoc));
      }
    }

    mlir::Value lhsTuple = builder().create<EquationSideOp>(location, lhs);
    mlir::Value rhsTuple = builder().create<EquationSideOp>(location, rhs);
    builder().create<EquationSidesOp>(location, lhsTuple, rhsTuple);
    builder().setInsertionPointAfter(equationOp);
  }
}
