#include "marco/Codegen/Lowering/BaseModelica/IfStatementLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
IfStatementLowerer::IfStatementLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

bool IfStatementLowerer::lower(const ast::bmodelica::IfStatement &statement) {
  mlir::Location location = loc(statement.getLocation());

  std::optional<mlir::Value> ifCondition =
      lowerCondition(*statement.getIfCondition());
  if (!ifCondition) {
    return false;
  }

  auto ifOp = builder().create<IfOp>(location, *ifCondition,
                                     statement.hasElseIfBlocks() ||
                                         statement.hasElseBlock());

  builder().setInsertionPointToStart(ifOp.thenBlock());

  // Lower the statements belonging to the 'if' block.
  if (!lower(*statement.getIfBlock())) {
    return false;
  }

  // Lower the statements belonging to the 'else if' blocks.
  if (statement.hasElseIfBlocks() || statement.hasElseBlock()) {
    builder().setInsertionPointToStart(ifOp.elseBlock());

    for (size_t i = 0, e = statement.getNumOfElseIfBlocks(); i < e; ++i) {
      std::optional<mlir::Value> elseIfCondition =
          lowerCondition(*statement.getElseIfCondition(i));
      if (!elseIfCondition) {
        return false;
      }

      auto elseIfOp = builder().create<IfOp>(
          location, *elseIfCondition, i != e - 1 || statement.hasElseBlock());

      builder().setInsertionPointToStart(elseIfOp.thenBlock());
      if (!lower(*statement.getElseIfBlock(i))) {
        return false;
      }

      if (i != e - 1 || statement.hasElseBlock()) {
        builder().setInsertionPointToStart(elseIfOp.elseBlock());
      }
    }
  }

  // Lower the statements belonging to the 'else' block.
  if (statement.hasElseBlock()) {
    if (!lower(*statement.getElseBlock())) {
      return false;
    }
  }

  // Continue creating the operations after the outermost IfOp.
  builder().setInsertionPointAfter(ifOp);

  return true;
}

std::optional<mlir::Value> IfStatementLowerer::lowerCondition(
    const ast::bmodelica::Expression &expression) {
  mlir::Location location = loc(expression.getLocation());
  auto loweredExpression = lower(expression);
  if (!loweredExpression) {
    return std::nullopt;
  }
  auto &results = *loweredExpression;
  assert(results.size() == 1);
  return results[0].get(location);
}

bool IfStatementLowerer::lower(
    const ast::bmodelica::StatementsBlock &statementsBlock) {
  for (size_t i = 0, e = statementsBlock.size(); i < e; ++i) {
    if (!lower(*statementsBlock[i])) {
      return false;
    }
  }
  return true;
}
} // namespace marco::codegen::lowering::bmodelica
