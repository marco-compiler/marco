#ifndef MARCO_AST_NODE_EXPRESSION_H
#define MARCO_AST_NODE_EXPRESSION_H

#include "marco/AST/Node/ASTNode.h"

namespace marco::ast {
class Expression : public ASTNode {
public:
  using ASTNode::ASTNode;

  virtual ~Expression();

  static bool classof(const ASTNode *node) {
    return node->getKind() >= ASTNode::Kind::Expression &&
           node->getKind() <= ASTNode::Kind::Expression_LastExpression;
  }

  virtual bool isLValue() const = 0;
};
} // namespace marco::ast

#endif // MARCO_AST_NODE_EXPRESSION_H
