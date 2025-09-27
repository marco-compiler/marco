#ifndef MARCO_AST_BASEMODELICA_EXPRESSIONFUNCTIONCALLARGUMENT_H
#define MARCO_AST_BASEMODELICA_EXPRESSIONFUNCTIONCALLARGUMENT_H

#include "marco/AST/BaseModelica/FunctionArgument.h"

namespace marco::ast::bmodelica {
class Expression;

class ExpressionFunctionArgument : public FunctionArgument {
public:
  explicit ExpressionFunctionArgument(SourceRange location);

  ExpressionFunctionArgument(const ExpressionFunctionArgument &other);

  ~ExpressionFunctionArgument() override;

  static bool classof(const ASTNode *node) {
    return node->getKind<ASTNodeKind>() ==
           ASTNodeKind::FunctionArgument_Expression;
  }

  std::unique_ptr<ast::ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Expression *getExpression();

  const Expression *getExpression() const;

  void setExpression(std::unique_ptr<ASTNode> node);

private:
  std::unique_ptr<ASTNode> expression;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_EXPRESSIONFUNCTIONCALLARGUMENT_H
