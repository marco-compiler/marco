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

  void EquationLowerer::lower(const ast::Equation& equation)
  {
    mlir::Location location = loc(equation.getLocation());
    auto equationOp = builder().create<EquationOp>(location);
    mlir::OpBuilder::InsertionGuard guard(builder());
    assert(equationOp.bodyRegion().empty());
    mlir::Block* equationBodyBlock = builder().createBlock(&equationOp.bodyRegion());
    builder().setInsertionPointToStart(equationBodyBlock);

    llvm::SmallVector<mlir::Value, 1> lhs;
    llvm::SmallVector<mlir::Value, 1> rhs;

    {
      // Left-hand side
      const auto* expression = equation.getLhsExpression();
      auto references = lower(*expression);

      for (auto& reference : references) {
        lhs.push_back(*reference);
      }
    }

    {
      // Right-hand side
      const auto* expression = equation.getRhsExpression();
      auto references = lower(*expression);

      for (auto& reference : references) {
        rhs.push_back(*reference);
      }
    }

    mlir::Value lhsTuple = builder().create<EquationSideOp>(location, lhs);
    mlir::Value rhsTuple = builder().create<EquationSideOp>(location, rhs);
    builder().create<EquationSidesOp>(location, lhsTuple, rhsTuple);
  }

  void EquationLowerer::lower(const ast::ForEquation& forEquation)
  {
    Lowerer::SymbolScope varScope(symbolTable());
    mlir::Location location = loc(forEquation.getEquation()->getLocation());

    // We need to keep track of the first loop in order to restore
    // the insertion point right after that when we have finished
    // lowering all the nested inductions.
    mlir::Operation* firstOp = nullptr;

    llvm::SmallVector<mlir::Value, 3> inductions;

    for (auto& induction : forEquation.getInductions())
    {
      const auto& startExpression = induction->getBegin();
      assert(startExpression->isa<Constant>());
      long start = startExpression->get<Constant>()->as<BuiltInType::Integer>();

      const auto& endExpression = induction->getEnd();
      assert(endExpression->isa<Constant>());
      long end = endExpression->get<Constant>()->as<BuiltInType::Integer>();

      auto forEquationOp = builder().create<ForEquationOp>(
          location, builder().getIndexAttr(start), builder().getIndexAttr(end));

      assert(forEquationOp.bodyRegion().getBlocks().empty());
      builder().createBlock(&forEquationOp.bodyRegion(), {}, builder().getIndexType());
      builder().setInsertionPointToStart(forEquationOp.bodyBlock());

      symbolTable().insert(
          induction->getName(),
          Reference::ssa(&builder(), forEquationOp.induction()));

      if (firstOp == nullptr) {
        firstOp = forEquationOp.getOperation();
      }
    }

    lower(*forEquation.getEquation());
    builder().setInsertionPointAfter(firstOp);
  }
}
