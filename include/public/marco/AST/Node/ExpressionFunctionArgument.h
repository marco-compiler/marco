#ifndef PUBLIC_MARCO_AST_NODE_EXPRESSIONFUNCTIONARGUMENT_H
#define PUBLIC_MARCO_AST_NODE_EXPRESSIONFUNCTIONARGUMENT_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/FunctionArgument.h"
#include "marco/Parser/Location.h"
#include <llvm/Support/JSON.h>
#include <memory>

namespace marco::ast {
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
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_EXPRESSIONFUNCTIONARGUMENT_H
