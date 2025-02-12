#ifndef MARCO_AST_NODE_REDUCTIONFUNCTIONCALLARGUMENT_H
#define MARCO_AST_NODE_REDUCTIONFUNCTIONCALLARGUMENT_H

#include "marco/AST/Node/FunctionArgument.h"

namespace marco::ast {
class Expression;
class ForIndex;

class ReductionFunctionArgument : public FunctionArgument {
public:
  ReductionFunctionArgument(SourceRange location);

  ReductionFunctionArgument(const ReductionFunctionArgument &other);

  ~ReductionFunctionArgument() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::FunctionArgument_Reduction;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Expression *getExpression();

  const Expression *getExpression() const;

  void setExpression(std::unique_ptr<ASTNode> node);

  size_t getNumOfForIndices() const;

  ForIndex *getForIndex(size_t index);

  const ForIndex *getForIndex(size_t index) const;

  llvm::ArrayRef<std::unique_ptr<ASTNode>> getForIndices() const;

  void setForIndices(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

private:
  std::unique_ptr<ASTNode> expression;
  llvm::SmallVector<std::unique_ptr<ASTNode>> forIndices;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_REDUCTIONFUNCTIONCALLARGUMENT_H
