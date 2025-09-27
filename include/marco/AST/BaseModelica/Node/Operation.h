#ifndef MARCO_AST_BASEMODELICA_NODE_OPERATION_H
#define MARCO_AST_BASEMODELICA_NODE_OPERATION_H

#include "marco/AST/BaseModelica/Node/Expression.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast::bmodelica {
enum class OperationKind {
  unknown,
  negate,
  add,
  addEW,
  subtract,
  subtractEW,
  multiply,
  multiplyEW,
  divide,
  divideEW,
  ifelse,
  greater,
  greaterEqual,
  equal,
  different,
  lessEqual,
  less,
  land,
  lor,
  lnot,
  subscription,
  powerOf,
  powerOfEW,
  range,
};

class Operation : public Expression {
public:
  explicit Operation(SourceRange location);

  Operation(const Operation &other);

  ~Operation() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Expression_Operation;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  bool isLValue() const override;

  OperationKind getOperationKind() const;

  void setOperationKind(OperationKind newOperationKind);

  size_t getNumOfArguments() const;

  Expression *getArgument(size_t index);

  const Expression *getArgument(size_t index) const;

  llvm::ArrayRef<std::unique_ptr<ASTNode>> getArguments() const;

  void setArguments(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  OperationKind kind{OperationKind::unknown};
  llvm::SmallVector<std::unique_ptr<ASTNode>, 3> arguments;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_OPERATION_H
