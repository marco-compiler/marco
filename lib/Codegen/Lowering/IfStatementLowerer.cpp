#include "marco/Codegen/Lowering/IfStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering
{
  IfStatementLowerer::IfStatementLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  void IfStatementLowerer::lower(const ast::IfStatement& statement)
  {
    mlir::Location location = loc(statement.getLocation());

    mlir::Value ifCondition = lowerCondition(*statement.getIfCondition());

    auto ifOp = builder().create<IfOp>(
        location, ifCondition,
        statement.hasElseIfBlocks() || statement.hasElseBlock());

    builder().setInsertionPointToStart(ifOp.thenBlock());

    // Lower the statements belonging to the 'if' block.
    lower(*statement.getIfBlock());

    // Lower the statements belonging to the 'else if' blocks.
    if (statement.hasElseIfBlocks() || statement.hasElseBlock()) {
      builder().setInsertionPointToStart(ifOp.elseBlock());

      for (size_t i = 0, e = statement.getNumOfElseIfBlocks(); i < e; ++i) {
        mlir::Value elseIfCondition =
            lowerCondition(*statement.getElseIfCondition(i));

        auto elseIfOp = builder().create<IfOp>(
            location, elseIfCondition,
            i != e - 1 || statement.hasElseBlock());

        builder().setInsertionPointToStart(elseIfOp.thenBlock());
        lower(*statement.getElseIfBlock(i));

        if (i != e - 1 || statement.hasElseBlock()) {
          builder().setInsertionPointToStart(elseIfOp.elseBlock());
        }
      }
    }

    // Lower the statements belonging to the 'else' block.
    if (statement.hasElseBlock()) {
      lower(*statement.getElseBlock());
    }

    // Continue creating the operations after the outermost IfOp.
    builder().setInsertionPointAfter(ifOp);
  }

  mlir::Value IfStatementLowerer::lowerCondition(
      const ast::Expression& expression)
  {
    mlir::Location location = loc(expression.getLocation());
    auto results = lower(expression);
    assert(results.size() == 1);
    return results[0].get(location);
  }

  void IfStatementLowerer::lower(const ast::StatementsBlock& statementsBlock)
  {
    for (size_t i = 0, e = statementsBlock.size(); i < e; ++i) {
      lower(*statementsBlock[i]);
    }
  }
}
