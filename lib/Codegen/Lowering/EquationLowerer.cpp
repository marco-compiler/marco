#include "marco/Codegen/Lowering/EquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  EquationLowerer::EquationLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void EquationLowerer::lower(
      const ast::Equation& equation, bool initialEquation)
  {
    mlir::Location location = loc(equation.getLocation());
    mlir::Operation* op = nullptr;

    if (initialEquation) {
      auto initialEquationOp = builder().create<InitialEquationOp>(location);
      op = initialEquationOp.getOperation();
      assert(initialEquationOp.getBodyRegion().empty());
      mlir::Block* bodyBlock =
          builder().createBlock(&initialEquationOp.getBodyRegion());
      builder().setInsertionPointToStart(bodyBlock);
    } else {
      auto equationOp = builder().create<EquationOp>(location);
      op = equationOp.getOperation();
      assert(equationOp.getBodyRegion().empty());

      mlir::Block* bodyBlock =
          builder().createBlock(&equationOp.getBodyRegion());

      builder().setInsertionPointToStart(bodyBlock);
    }

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
    builder().setInsertionPointAfter(op);
  }

  void EquationLowerer::lower(
      const ast::ForEquation& forEquation, bool initialEquation)
  {
    Lowerer::VariablesScope scope(getVariablesSymbolTable());
    mlir::Location location = loc(forEquation.getEquation()->getLocation());

    // We need to keep track of the first loop in order to restore
    // the insertion point right after that when we have finished
    // lowering all the nested inductions.
    mlir::Operation* firstOp = nullptr;

    llvm::SmallVector<mlir::Value, 3> inductions;

    for (size_t i = 0, e = forEquation.getNumOfInductions(); i < e; ++i) {
      const ast::Induction* induction = forEquation.getInduction(i);

      const ast::Expression* startExpression = induction->getBegin();
      assert(startExpression->isa<ast::Constant>());
      long start = startExpression->cast<ast::Constant>()->as<int64_t>();

      const ast::Expression* endExpression = induction->getEnd();
      assert(endExpression->isa<ast::Constant>());
      long end = endExpression->cast<ast::Constant>()->as<int64_t>();

      const ast::Expression* stepExpression = induction->getStep();
      assert(stepExpression->isa<ast::Constant>());
      long step = stepExpression->cast<ast::Constant>()->as<int64_t>();

      auto forEquationOp =
          builder().create<ForEquationOp>(location, start, end, step);

      builder().setInsertionPointToStart(forEquationOp.bodyBlock());

      // Add the induction variable to the symbol table
      getVariablesSymbolTable().insert(
          induction->getName(),
          Reference::ssa(builder(), forEquationOp.induction()));

      if (firstOp == nullptr) {
        firstOp = forEquationOp.getOperation();
      }
    }

    // Create the equation body.
    lower(*forEquation.getEquation(), initialEquation);

    builder().setInsertionPointAfter(firstOp);
  }
}
