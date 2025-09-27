#ifndef MARCO_AST_BASEMODELICA_NODE_SUBSCRIPT_H
#define MARCO_AST_BASEMODELICA_NODE_SUBSCRIPT_H

#include "marco/AST/BaseModelica/Node/Expression.h"

namespace marco::ast::bmodelica {
// Following the official Modelica grammar specification, Subscript should
// not be an expression. However, Flat Modelica requires it to be.
class Subscript : public ASTNode {
public:
  Subscript(SourceRange location);

  Subscript(const Subscript &other);

  ~Subscript() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Expression_Subscript;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  bool isUnbounded() const;

  Expression *getExpression();

  const Expression *getExpression() const;

  void setExpression(std::unique_ptr<ASTNode> node);

private:
  std::unique_ptr<ASTNode> expression;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_SUBSCRIPT_H
