#include "marco/Codegen/Lowering/ForEquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ForEquationLowerer::ForEquationLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void ForEquationLowerer::lower(const ast::ForEquation& forEquation)
  {
    Lowerer::VariablesScope scope(getVariablesSymbolTable());
    mlir::Location location = loc(forEquation.getLocation());

    // We need to keep track of the first loop in order to restore
    // the insertion point right after that when we have finished
    // lowering all the nested inductions.
    mlir::Operation* firstOp = nullptr;

    llvm::SmallVector<mlir::Value, 3> inductions;

    for (size_t i = 0, e = forEquation.getNumOfForIndices(); i < e; ++i) {
      const ast::ForIndex* forIndex = forEquation.getForIndex(i);
      assert(forIndex->hasExpression());
      assert(forIndex->getExpression()->isa<ast::Operation>());

      const auto* forIndexRange =
          forIndex->getExpression()->cast<ast::Operation>();

      assert(forIndexRange->getOperationKind() == ast::OperationKind::range);

      int64_t begin, end, step;

      if (forIndexRange->getNumOfArguments() == 2) {
        assert(forIndexRange->getArgument(0)->isa<ast::Constant>());
        assert(forIndexRange->getArgument(1)->isa<ast::Constant>());

        begin = forIndexRange->getArgument(0)
                    ->dyn_cast<ast::Constant>()->as<int64_t>();

        step = 1;

        end = forIndexRange->getArgument(1)
                  ->dyn_cast<ast::Constant>()->as<int64_t>();
      } else {
        assert(forIndexRange->getArgument(0)->isa<ast::Constant>());
        assert(forIndexRange->getArgument(1)->isa<ast::Constant>());
        assert(forIndexRange->getArgument(2)->isa<ast::Constant>());

        begin = forIndexRange->getArgument(0)
                    ->dyn_cast<ast::Constant>()->as<int64_t>();

        step = forIndexRange->getArgument(1)
                    ->dyn_cast<ast::Constant>()->as<int64_t>();

        end = forIndexRange->getArgument(2)
                    ->dyn_cast<ast::Constant>()->as<int64_t>();
      }

      auto forEquationOp =
          builder().create<ForEquationOp>(location, begin, end, step);

      builder().setInsertionPointToStart(forEquationOp.bodyBlock());

      // Add the induction variable to the symbol table
      getVariablesSymbolTable().insert(
          forIndex->getName(),
          Reference::ssa(builder(), forEquationOp.induction()));

      if (firstOp == nullptr) {
        firstOp = forEquationOp.getOperation();
      }
    }

    // Create the equation body.
    for (size_t i = 0, e = forEquation.getNumOfEquations(); i < e; ++i) {
      lower(*forEquation.getEquation(i));
    }

    builder().setInsertionPointAfter(firstOp);
  }
}
