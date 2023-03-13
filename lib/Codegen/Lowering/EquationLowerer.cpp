#include "marco/Codegen/Lowering/EquationLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  EquationLowerer::EquationLowerer(LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge)
  {
  }

  void EquationLowerer::lower(const ast::Equation& equation, bool initialEquation)
  {
    mlir::Location location = loc(equation.getLocation());
    mlir::Operation* op = nullptr;

    if (initialEquation) {
      auto initialEquationOp = builder().create<InitialEquationOp>(location);
      op = initialEquationOp.getOperation();
      assert(initialEquationOp.getBodyRegion().empty());
      mlir::Block* bodyBlock = builder().createBlock(&initialEquationOp.getBodyRegion());
      builder().setInsertionPointToStart(bodyBlock);
    } else {
      auto equationOp = builder().create<EquationOp>(location);
      op = equationOp.getOperation();
      assert(equationOp.getBodyRegion().empty());
      mlir::Block* bodyBlock = builder().createBlock(&equationOp.getBodyRegion());
      builder().setInsertionPointToStart(bodyBlock);
    }

    llvm::SmallVector<mlir::Value, 1> lhs;
    llvm::SmallVector<mlir::Value, 1> rhs;

    {
      // Left-hand side
      const auto* expression = equation.getLhsExpression();

      auto referencesLoc = loc(expression->getLocation());
      auto references = lower(*expression);

      for (auto& reference : references) {
        lhs.push_back(reference.get(referencesLoc));
      }
    }

    {
      // Right-hand side
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

  void EquationLowerer::lower(const ast::ForEquation& forEquation, bool initialEquation)
  {
    Lowerer::SymbolScope scope(symbolTable());
    mlir::Location location = loc(forEquation.getEquation()->getLocation());

    // We need to keep track of the first loop in order to restore
    // the insertion point right after that when we have finished
    // lowering all the nested inductions.
    mlir::Operation* firstOp = nullptr;

    llvm::SmallVector<mlir::Value, 3> inductions;

    for (auto& induction : forEquation.getInductions()) {
      const auto& startExpression = induction->getBegin();
      assert(startExpression->isa<Constant>());
      long start = startExpression->get<Constant>()->as<BuiltInType::Integer>();

      const auto& endExpression = induction->getEnd();
      assert(endExpression->isa<Constant>());
      long end = endExpression->get<Constant>()->as<BuiltInType::Integer>();

      const auto& stepExpression = induction->getStep();
      assert(stepExpression->isa<Constant>());
      long step = stepExpression->get<Constant>()->as<BuiltInType::Integer>();

      auto forEquationOp = builder().create<ForEquationOp>(location, start, end, step);
      builder().setInsertionPointToStart(forEquationOp.bodyBlock());

      // Add the induction variable to the symbol table
      symbolTable().insert(
          induction->getName(),
          Reference::ssa(builder(), forEquationOp.induction()));

      if (firstOp == nullptr) {
        firstOp = forEquationOp.getOperation();
      }
    }

    // Create the equation body
    lower(*forEquation.getEquation(), initialEquation);

    builder().setInsertionPointAfter(firstOp);
  }
}
