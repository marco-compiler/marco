#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_INDEXEXPRESSION_INDEXEXPRESSION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_INDEXEXPRESSION_INDEXEXPRESSION_H

#include <memory>

// Currently header-only.

namespace mlir::bmodelica {
struct IndexExpressionNode {
  enum class NodeType {
    ROOT,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    // Assumption -- always int64_t representable
    // Functions as a placeholder we can insert loop bounds into later
    VALUE,
    CONSTANT
  };

  NodeType type;

  std::unique_ptr<IndexExpressionNode> left;
  std::unique_ptr<IndexExpressionNode> right;
};

struct IndexExpression {
  IndexExpressionNode root;
  size_t numParams;
};
} // namespace mlir::bmodelica


#endif
