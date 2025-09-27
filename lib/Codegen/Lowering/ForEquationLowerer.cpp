#include "marco/Codegen/Lowering/BaseModelica/ForEquationLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
ForEquationLowerer::ForEquationLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool ForEquationLowerer::lower(const ast::bmodelica::ForEquation &forEquation) {
  VariablesSymbolTable::VariablesScope scope(getVariablesSymbolTable());
  mlir::Location location = loc(forEquation.getLocation());

  // We need to keep track of the first loop in order to restore
  // the insertion point right after that when we have finished
  // lowering all the nested inductions.
  mlir::Operation *firstOp = nullptr;

  llvm::SmallVector<mlir::Value, 3> inductions;

  for (size_t i = 0, e = forEquation.getNumOfForIndices(); i < e; ++i) {
    const ast::bmodelica::ForIndex *forIndex = forEquation.getForIndex(i);
    assert(forIndex->hasExpression());
    assert(forIndex->getExpression()->isa<ast::bmodelica::Operation>());

    const auto *forIndexRange =
        forIndex->getExpression()->cast<ast::bmodelica::Operation>();

    assert(forIndexRange->getOperationKind() ==
           ast::bmodelica::OperationKind::range);

    int64_t begin, end, step;

    if (forIndexRange->getNumOfArguments() == 2) {
      assert(forIndexRange->getArgument(0)->isa<ast::bmodelica::Constant>());
      assert(forIndexRange->getArgument(1)->isa<ast::bmodelica::Constant>());

      begin = forIndexRange->getArgument(0)
                  ->dyn_cast<ast::bmodelica::Constant>()
                  ->as<int64_t>();

      step = 1;

      end = forIndexRange->getArgument(1)
                ->dyn_cast<ast::bmodelica::Constant>()
                ->as<int64_t>();
    } else {
      assert(forIndexRange->getArgument(0)->isa<ast::bmodelica::Constant>());
      assert(forIndexRange->getArgument(1)->isa<ast::bmodelica::Constant>());
      assert(forIndexRange->getArgument(2)->isa<ast::bmodelica::Constant>());

      begin = forIndexRange->getArgument(0)
                  ->dyn_cast<ast::bmodelica::Constant>()
                  ->as<int64_t>();

      step = forIndexRange->getArgument(1)
                 ->dyn_cast<ast::bmodelica::Constant>()
                 ->as<int64_t>();

      end = forIndexRange->getArgument(2)
                ->dyn_cast<ast::bmodelica::Constant>()
                ->as<int64_t>();
    }

    auto forEquationOp =
        builder().create<ForEquationOp>(location, begin, end, step);

    builder().setInsertionPointToStart(forEquationOp.bodyBlock());

    // Add the induction variable to the symbol table
    llvm::StringRef name = forIndex->getName();
    getVariablesSymbolTable().insert(
        name, Reference::ssa(builder(), forEquationOp.induction()));

    if (firstOp == nullptr) {
      firstOp = forEquationOp.getOperation();
    }
  }

  // Create the equation body.
  for (size_t i = 0, e = forEquation.getNumOfEquations(); i < e; ++i) {
    if (!lower(*forEquation.getEquation(i))) {
      return false;
    }
  }

  builder().setInsertionPointAfter(firstOp);

  return true;
}
} // namespace marco::codegen::lowering::bmodelica
