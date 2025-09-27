#ifndef MARCO_AST_BASEMODELICA_NODE_EXPRESSIONFUNCTIONCALLARGUMENT_H
#define MARCO_AST_BASEMODELICA_NODE_EXPRESSIONFUNCTIONCALLARGUMENT_H

#include "marco/AST/BaseModelica/Node/FunctionArgument.h"

namespace marco::ast::bmodelica {
class Expression;

class ExpressionFunctionArgument : public FunctionArgument {
public:
  explicit ExpressionFunctionArgument(SourceRange location);

  ExpressionFunctionArgument(const ExpressionFunctionArgument &other);

  ~ExpressionFunctionArgument() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::FunctionArgument_Expression;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Expression *getExpression();

  const Expression *getExpression() const;

  void setExpression(std::unique_ptr<ASTNode> node);

private:
  std::unique_ptr<ASTNode> expression;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_EXPRESSIONFUNCTIONCALLARGUMENT_H
