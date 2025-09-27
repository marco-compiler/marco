#ifndef MARCO_AST_BASEMODELICA_NODE_STATEMENT_H
#define MARCO_AST_BASEMODELICA_NODE_STATEMENT_H

#include "marco/AST/BaseModelica/Node/ASTNode.h"

namespace marco::ast::bmodelica {
class Statement : public ASTNode {
public:
  using ASTNode::ASTNode;

  Statement(const Statement &other);

  ~Statement() override;

  static bool classof(const ASTNode *node) {
    return node->getKind() >= ASTNode::Kind::Statement &&
           node->getKind() <= ASTNode::Kind::Statement_LastStatement;
  }
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_STATEMENT_H
