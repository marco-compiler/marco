#ifndef MARCO_AST_BASEMODELICA_NODE_EXPRESSION_H
#define MARCO_AST_BASEMODELICA_NODE_EXPRESSION_H

#include "marco/AST/BaseModelica/Node/ASTNode.h"

namespace marco::ast::bmodelica {
class Expression : public ASTNode {
public:
  using ASTNode::ASTNode;

  ~Expression() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() >= ASTNode::Kind::Expression &&
           node->getKind() <= ASTNode::Kind::Expression_LastExpression;
  }

  virtual bool isLValue() const = 0;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_EXPRESSION_H
