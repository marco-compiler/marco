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

  __attribute__((warn_unused_result)) bool 
  IfStatementLowerer::lower(const ast::IfStatement& statement)
  {
    mlir::Location location = loc(statement.getLocation());

    std::optional<mlir::Value> optionalIfCondition = 
        lowerCondition(*statement.getIfCondition());
    if (!optionalIfCondition) {
      return false;
    }
    mlir::Value &ifCondition = optionalIfCondition.value();

    auto ifOp = builder().create<IfOp>(
        location, ifCondition,
        statement.hasElseIfBlocks() || statement.hasElseBlock());

    builder().setInsertionPointToStart(ifOp.thenBlock());

    // Lower the statements belonging to the 'if' block.
    bool outcome = lower(*statement.getIfBlock());
    if (!outcome) {
      return false;
    }

    // Lower the statements belonging to the 'else if' blocks.
    if (statement.hasElseIfBlocks() || statement.hasElseBlock()) {
      builder().setInsertionPointToStart(ifOp.elseBlock());

      for (size_t i = 0, e = statement.getNumOfElseIfBlocks(); i < e; ++i) {
        std::optional<mlir::Value> optionalElseIfCondition = 
            lowerCondition(*statement.getElseIfCondition(i));
        if (!optionalElseIfCondition) {
          return false;
        }
        mlir::Value &elseIfCondition = optionalElseIfCondition.value();

        auto elseIfOp = builder().create<IfOp>(
            location, elseIfCondition,
            i != e - 1 || statement.hasElseBlock());

        builder().setInsertionPointToStart(elseIfOp.thenBlock());
        outcome = lower(*statement.getElseIfBlock(i));
        if (!outcome) {
          return false;
        }

        if (i != e - 1 || statement.hasElseBlock()) {
          builder().setInsertionPointToStart(elseIfOp.elseBlock());
        }
      }
    }

    // Lower the statements belonging to the 'else' block.
    if (statement.hasElseBlock()) {
      outcome = lower(*statement.getElseBlock());
      if (!outcome) {
        return false;
      }
    }

    // Continue creating the operations after the outermost IfOp.
    builder().setInsertionPointAfter(ifOp);

    return true;
  }

  std::optional<mlir::Value> IfStatementLowerer::lowerCondition(
      const ast::Expression& expression)
  {
    mlir::Location location = loc(expression.getLocation());
    auto optionalResults = lower(expression);
    if (!optionalResults) {
      return std::nullopt;
    }
    auto &results = optionalResults.value();
    assert(results.size() == 1);
    return results[0].get(location);
  }

  __attribute__((warn_unused_result)) bool
  IfStatementLowerer::lower(const ast::StatementsBlock& statementsBlock)
  {
    for (size_t i = 0, e = statementsBlock.size(); i < e; ++i) {
      const bool outcome = lower(*statementsBlock[i]);
      if (!outcome) {
        return false;
      }
    }
    return true;
  }
}
